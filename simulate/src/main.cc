// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic> // Required for std::atomic
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <vector> // Required for std::vector used in ElasticBand

#include "unitree_sdk2_bridge/unitree_sdk2_bridge.h" // Assuming this header is correct
#include "yaml-cpp/yaml.h"
#include <mujoco/mujoco.h>
#include <pthread.h>

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"

extern "C" {
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::sample_util;

const int kErrorLength = 1024; // load error string length

// model and data
mjModel *m = nullptr;
mjData *d = nullptr;

// control noise variables
mjtNum *ctrlnoise = nullptr;

std::atomic<bool> exit_request = false; // Global flag to signal exit

struct SimulationConfig {
  std::string robot = "go2";
  std::string robot_scene = "scene.xml";

  int domain_id = 1;
  std::string interface = "lo";

  int use_joystick = 0;
  std::string joystick_type = "xbox";
  std::string joystick_device = "/dev/input/js0";
  int joystick_bits = 16;

  int print_scene_information = 1;

  int enable_elastic_band = 0;
  // int band_attached_link = 0; // This will be determined by name now

  double ctrl_noise_std = 0.0;
  double ctrl_noise_rate = 0.1;
} config;

namespace headless_sim {
struct ElasticBand {
  bool enable_ = false;
  double f_[3] = {0.0, 0.0, 0.0};
  double k_stiffness_ = 100.0;
  double x0_equilibrium_[3] = {0.0, 0.0, 0.0};
  double damping_coeff_ = 10.0;
  int attached_body_id_ = -1; // Store the body ID for the elastic band

  void Init(bool enabled, double stiffness, const double equilibrium[3],
            double damping, mjModel *model, const std::string &body_name) {
    enable_ = enabled;
    if (!enable_)
      return;

    k_stiffness_ = stiffness;
    x0_equilibrium_[0] = equilibrium[0];
    x0_equilibrium_[1] = equilibrium[1];
    x0_equilibrium_[2] = equilibrium[2];
    damping_coeff_ = damping;

    if (model && !body_name.empty()) {
      attached_body_id_ = mj_name2id(model, mjOBJ_BODY, body_name.c_str());
      if (attached_body_id_ == -1) {
        std::cerr << "ElasticBand Warning: Could not find body named '"
                  << body_name << "'" << std::endl;
        enable_ = false; // Disable if body not found
      } else {
        std::cout << "ElasticBand initialized for body '" << body_name
                  << "' (ID: " << attached_body_id_ << ")" << std::endl;
      }
    } else {
      enable_ = false; // Disable if model or body name is missing
      std::cerr << "ElasticBand Warning: Model or body name not provided for "
                   "initialization."
                << std::endl;
    }
  }

  void Advance(const mjData *data_ptr) { // Pass mjData to get body pose
    if (!enable_ || attached_body_id_ == -1 || !data_ptr) {
      f_[0] = f_[1] = f_[2] = 0.0;
      return;
    }

    // Get current position and velocity of the attached body's CoM in world
    // frame xpos is [body_id * 3 + (0,1,2)] cvel is [body_id * 6 + (3,4,5 for
    // linear vel)]
    std::vector<double> x_current(3), dx_current(3);
    for (int i = 0; i < 3; ++i) {
      x_current[i] =
          data_ptr->xipos[attached_body_id_ * 3 + i]; // Center of mass position
      dx_current[i] =
          data_ptr
              ->cvel[attached_body_id_ * 6 + (3 + i)]; // Linear velocity of CoM
    }

    // Simplified spring-damper logic
    for (int i = 0; i < 3; ++i) {
      f_[i] = -k_stiffness_ * (x_current[i] - x0_equilibrium_[i]) -
              damping_coeff_ * dx_current[i];
    }
  }

  void ApplyForce(mjData *data_ptr) {
    if (!enable_ || attached_body_id_ == -1 || !data_ptr) {
      return;
    }
    // Apply calculated force to the center of mass of the attached body
    // xfrc_applied is [body_id * 6 + (0,1,2 for force)]
    for (int i = 0; i < 3; ++i) {
      data_ptr->xfrc_applied[attached_body_id_ * 6 + i] +=
          f_[i]; // Add force, allows other forces
    }
    // Torque is not applied by this simple band
  }
};
ElasticBand elastic_band_instance;
} // namespace headless_sim

using Seconds = std::chrono::duration<double>;

std::string getExecutableDir() {
#if defined(_WIN32) || defined(__CYGWIN__)
  constexpr char kPathSep = '\\';
  std::string realpath_str = [&]() -> std::string {
    std::unique_ptr<char[]> realpath_ptr(nullptr);
    DWORD buf_size = 128;
    bool success = false;
    while (!success) {
      realpath_ptr.reset(new (std::nothrow) char[buf_size]);
      if (!realpath_ptr) {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }

      DWORD written = GetModuleFileNameA(nullptr, realpath_ptr.get(), buf_size);
      if (written < buf_size) {
        success = true;
      } else if (written == buf_size) {
        buf_size *= 2;
      } else {
        std::cerr << "failed to retrieve executable path: " << GetLastError()
                  << "\n";
        return "";
      }
    }
    return realpath_ptr.get();
  }();
#else // Not Windows
  constexpr char kPathSep = '/';
  const char *path_for_readlink; // Will point to the path to be resolved

#if defined(__APPLE__)
  std::unique_ptr<char[]> buf(
      nullptr); // Needs to stay in scope for path_for_readlink
  {
    std::uint32_t temp_buf_size = 0;
    _NSGetExecutablePath(nullptr, &temp_buf_size);
    buf.reset(new char[temp_buf_size]);
    if (!buf) {
      std::cerr << "cannot allocate memory to store executable path (Apple)\n";
      return "";
    }
    if (_NSGetExecutablePath(buf.get(), &temp_buf_size) != 0) // 0 on success
    {
      // This case should ideally not happen if pre-flighting temp_buf_size
      // worked.
      std::cerr
          << "unexpected error from _NSGetExecutablePath or buffer too small\n";
      // return ""; // Or try to proceed if path is partially written, though
      // risky.
    }
  }
  path_for_readlink = buf.get();
#else // Not Apple (e.g., Linux)
  path_for_readlink = "/proc/self/exe";
#endif

  std::string realpath_str = [&]() -> std::string {
    std::unique_ptr<char[]> realpath_ptr(nullptr);
    std::uint32_t current_buf_size = 128; // Initial buffer size for readlink
    bool success = false;
    while (!success) {
      realpath_ptr.reset(new (std::nothrow) char[current_buf_size]);
      if (!realpath_ptr) {
        std::cerr
            << "cannot allocate memory to store executable path (readlink)\n";
        return "";
      }

      std::size_t written =
          readlink(path_for_readlink, realpath_ptr.get(), current_buf_size);

      if (written < current_buf_size) // Success, including case where written
                                      // == -1 (error)
      {
        if (written == static_cast<std::size_t>(-1)) // readlink error
        {
#if defined(__APPLE__)
          // On macOS, if path_for_readlink was from _NSGetExecutablePath,
          // it's already resolved. readlink might fail with EINVAL if it's not
          // a symlink.
          if (errno == EINVAL) {
            return path_for_readlink; // Use the path from _NSGetExecutablePath
                                      // directly
          }
#endif
          // General readlink error
          std::cerr << "error while resolving executable path with readlink: "
                    << strerror(errno) << '\n';
          return "";
        }
        // readlink success
        realpath_ptr.get()[written] = '\0'; // Null-terminate
        success = true;
      } else // Buffer too small (written == current_buf_size)
      {
        current_buf_size *= 2;
      }
    }
    return realpath_ptr.get();
  }();
#endif // End OS-specific path gathering

  if (realpath_str.empty()) {
    return "";
  }

  // Find the last path separator
  std::size_t last_sep = realpath_str.rfind(kPathSep);
  if (last_sep != std::string::npos) {
    return realpath_str.substr(0, last_sep);
  }

  // If no separator found, it might be a file in the current dir or root.
  // This behavior might need adjustment based on expectations.
  return ""; // Or return "." for current directory if appropriate
}

void scanPluginLibraries() {
  int nplugin = mjp_pluginCount();
  if (nplugin) {
    std::printf("Built-in plugins:\n");
    for (int i = 0; i < nplugin; ++i) {
      std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
    }
  }

#if defined(_WIN32) || defined(__CYGWIN__)
  const std::string sep = "\\";
#else
  const std::string sep = "/";
#endif

  const std::string executable_dir = getExecutableDir();
  if (executable_dir.empty()) {
    std::cerr << "Warning: Could not determine executable directory. Plugin "
                 "scanning might fail."
              << std::endl;
    return;
  }

  const std::string plugin_dir = executable_dir + sep + MUJOCO_PLUGIN_DIR;
  mj_loadAllPluginLibraries(
      plugin_dir.c_str(), +[](const char *filename, int first, int count) {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        }
      });
}

mjModel *LoadModel(const char *file) {
  char filename_arr[1024];
  mju::strcpy_arr(filename_arr, file);

  if (!filename_arr[0]) {
    std::cerr << "LoadModel error: Empty filename provided." << std::endl;
    return nullptr;
  }

  char loadError[kErrorLength] = "";
  mjModel *mnew = nullptr;
  if (mju::strlen_arr(filename_arr) > 4 &&
      !std::strncmp(filename_arr + mju::strlen_arr(filename_arr) - 4, ".mjb",
                    4)) {
    mnew = mj_loadModel(filename_arr, nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename_arr, nullptr, loadError, kErrorLength);
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (error_length > 0 && loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  if (!mnew) {
    std::fprintf(stderr, "LoadModel error: %s (file: %s)\n", loadError,
                 filename_arr);
    return nullptr;
  }

  if (loadError[0]) {
    std::printf("Model compiled, but simulation warning:\n  %s\n", loadError);
  }
  return mnew;
}

void PhysicsLoop() {
  // Elastic band is initialized in PhysicsThread after model is loaded.
  // headless_sim::elastic_band_instance.enable_ = (config.enable_elastic_band
  // == 1);

  while (!exit_request.load()) {
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    if (m && d) {
      // Clear previous external forces (important if not set every step or if
      // bridge also sets them) mju_zero(d->xfrc_applied, m->nbody * 6); //
      // Optional: Clear if you want only band forces

      if (config.ctrl_noise_std > 0.0 && m->nu > 0) {
        mjtNum rate = mju_exp(-m->opt.timestep /
                              mju_max(config.ctrl_noise_rate, mjMINVAL));
        mjtNum scale = config.ctrl_noise_std * mju_sqrt(1 - rate * rate);
        for (int i = 0; i < m->nu; i++) {
          ctrlnoise[i] =
              rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);
          // Assuming noise is additive to any control signal from the bridge
          // If bridge sets d->ctrl directly, this might overwrite or be
          // overwritten. A common pattern is for bridge to set a target, and a
          // low-level controller (or this noise) adds to it. Or, bridge writes
          // to a separate buffer, and then it's combined with noise into
          // d->ctrl. For now, directly setting d->ctrl with noise if no bridge
          // input is assumed. If bridge provides input, it should be d->ctrl[i]
          // = bridge_value[i] + ctrlnoise[i];
          d->ctrl[i] = ctrlnoise[i];
        }
      }

      if (config.enable_elastic_band == 1 &&
          headless_sim::elastic_band_instance.enable_) {
        headless_sim::elastic_band_instance.Advance(d); // Pass mjData
        headless_sim::elastic_band_instance.ApplyForce(
            d); // Apply the calculated force
      }

      mj_step(m, d);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  std::cout << "PhysicsLoop exiting." << std::endl;
}
} // namespace

void PhysicsThread(const char *filename_arg) {
  std::string scene_path_str;
  const char *effective_filename = filename_arg;

  if (filename_arg == nullptr || filename_arg[0] == '\0') {
    scene_path_str = "../../" + config.robot + "/" + config.robot_scene;
    effective_filename = scene_path_str.c_str();
    std::cout << "No filename provided via argument, using from config: "
              << effective_filename << std::endl;
  } else {
    std::cout << "Using filename from argument: " << effective_filename
              << std::endl;
  }

  if (effective_filename != nullptr && effective_filename[0] != '\0') {
    std::cout << "Loading model: " << effective_filename << std::endl;
    m = LoadModel(effective_filename);
    if (m) {
      d = mj_makeData(m);
    }

    if (m && d) {
      std::cout << "Model and data loaded successfully." << std::endl;
      mj_forward(m, d);

      if (ctrlnoise)
        free(ctrlnoise);
      if (m->nu > 0) { // Only allocate if there are controls
        ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      } else {
        ctrlnoise = nullptr;
      }

      // Initialize Elastic Band here, after model (m) is loaded
      if (config.enable_elastic_band == 1) {
        std::string target_body_name;
        if (config.robot == "h1" || config.robot == "g1") {
          target_body_name = "torso_link";
        } else { // Default for go2, etc.
          target_body_name = "base_link";
        }
        // Example equilibrium position (0,0,0) and stiffness/damping
        // These could also come from config.yaml
        double eq_pos[3] = {0.0, 0.0,
                            0.5}; // Example: Target 0.5m height for base_link
        headless_sim::elastic_band_instance.Init(true,   // enabled
                                                 50.0,   // stiffness
                                                 eq_pos, // equilibrium position
                                                 5.0,    // damping
                                                 m,      // mjModel pointer
                                                 target_body_name);
      }

    } else {
      std::cerr << "Failed to load model or make data from: "
                << effective_filename << std::endl;
      exit_request.store(true);
    }
  } else {
    std::cerr << "No model filename specified." << std::endl;
    exit_request.store(true);
  }

  if (!exit_request.load()) {
    PhysicsLoop();
  }

  if (ctrlnoise) {
    free(ctrlnoise);
    ctrlnoise = nullptr;
  }
  if (d) {
    mj_deleteData(d);
    d = nullptr;
  }
  if (m) {
    mj_deleteModel(m);
    m = nullptr;
  }
  std::cout << "PhysicsThread finished cleanup." << std::endl;
}

void *UnitreeSdk2BridgeThread(void *arg) {
  while (true) // Loop to check for exit request
  {
    if (exit_request.load()) {
      std::cout
          << "UnitreeSdk2BridgeThread: Exit request received during init wait."
          << std::endl;
      pthread_exit(NULL);
    }
    if (d && m) {
      std::cout << "Mujoco data is prepared for Unitree SDK Bridge."
                << std::endl;
      break;
    }
    usleep(500000); // 0.5 seconds
  }

  // Ensure m and d are valid before proceeding
  if (!m || !d) {
    std::cerr << "UnitreeSdk2BridgeThread: m or d is null. Exiting."
              << std::endl;
    pthread_exit(NULL);
  }

  // Elastic band related config.band_attached_link is now handled internally by
  // ElasticBand instance

  ChannelFactory::Instance()->Init(config.domain_id, config.interface);
  UnitreeSdk2Bridge unitree_interface(m, d);

  if (config.use_joystick == 1) {
    unitree_interface.SetupJoystick(config.joystick_device,
                                    config.joystick_type, config.joystick_bits);
  }

  if (config.print_scene_information == 1) {
    unitree_interface.PrintSceneInformation();
  }

  std::cout << "UnitreeSdk2BridgeThread: Starting Run()." << std::endl;
  // Modify UnitreeSdk2Bridge::Run() to be non-blocking or to periodically check
  // an exit flag For now, assuming it might block. If so, exit_request won't
  // stop it gracefully. A simple way if Run() is a loop: pass 'exit_request' to
  // it or make it a member. while(!exit_request.load()) {
  // unitree_interface.DoWork(); }
  unitree_interface.Run();
  std::cout << "UnitreeSdk2BridgeThread: Run() finished." << std::endl;

  pthread_exit(NULL);
}

#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void
_mj_rosettaError(const char *msg) {
  rosetta_error_msg = msg;
}
#endif

int main(int argc, char **argv) {
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }

  scanPluginLibraries();

  YAML::Node yaml_node;
  std::string config_file_path = "../config.yaml"; // Default path
  // Potentially allow config file path to be an argument in the future
  try {
    yaml_node = YAML::LoadFile(config_file_path);
    config.robot = yaml_node["robot"].as<std::string>(config.robot);
    config.robot_scene =
        yaml_node["robot_scene"].as<std::string>(config.robot_scene);
    config.domain_id = yaml_node["domain_id"].as<int>(config.domain_id);
    config.interface = yaml_node["interface"].as<std::string>(config.interface);
    config.print_scene_information =
        yaml_node["print_scene_information"].as<int>(
            config.print_scene_information);
    config.enable_elastic_band =
        yaml_node["enable_elastic_band"].as<int>(config.enable_elastic_band);
    config.use_joystick =
        yaml_node["use_joystick"].as<int>(config.use_joystick);
    config.joystick_type =
        yaml_node["joystick_type"].as<std::string>(config.joystick_type);
    config.joystick_device =
        yaml_node["joystick_device"].as<std::string>(config.joystick_device);
    config.joystick_bits =
        yaml_node["joystick_bits"].as<int>(config.joystick_bits);

    if (yaml_node["ctrl_noise_std"]) {
      config.ctrl_noise_std = yaml_node["ctrl_noise_std"].as<double>();
    }
    if (yaml_node["ctrl_noise_rate"]) {
      config.ctrl_noise_rate = yaml_node["ctrl_noise_rate"].as<double>();
    }
    std::cout << "Loaded configuration from " << config_file_path << std::endl;

  } catch (const YAML::Exception &e) {
    std::cerr << "Warning: Error loading or parsing " << config_file_path
              << ": " << e.what() << std::endl;
    std::cerr << "Using default/hardcoded configuration values." << std::endl;
  }

  const char *filename_main_arg = nullptr;
  if (argc > 1) {
    filename_main_arg = argv[1];
  }

  pthread_t unitree_thread_handle;
  int rc = pthread_create(&unitree_thread_handle, NULL, UnitreeSdk2BridgeThread,
                          NULL);
  if (rc != 0) {
    std::cerr << "Error: unable to create unitree_thread, rc: " << rc
              << std::endl;
    // No need to exit(-1) immediately, try to start physics thread anyway or
    // handle more gracefully. For now, we'll let it proceed and potentially
    // fail later if bridge is critical.
  }

  std::cout << "Starting Physics Thread..." << std::endl;
  std::thread physicsthreadhandle(&PhysicsThread, filename_main_arg);

  std::cout << "Simulation running headlessly. Press Ctrl+C to interrupt."
            << std::endl;

  // Main thread will now wait for physics thread to complete.
  // Physics thread will complete when exit_request is true.
  if (physicsthreadhandle.joinable()) {
    physicsthreadhandle.join();
  }
  std::cout << "Physics thread joined." << std::endl;

  // Ensure exit_request is set for the Unitree bridge thread if it wasn't
  // already
  exit_request.store(true);
  std::cout << "Requesting Unitree SDK Bridge thread to exit (if not already "
               "signaled)..."
            << std::endl;

  if (rc == 0) { // Only try to join if pthread_create succeeded
    // Give the Unitree thread a moment to react to exit_request before trying
    // to join. This is a HACK. UnitreeSdk2Bridge.Run() should be designed to be
    // interruptible. If it's a tight loop, it might not see exit_request in
    // time for pthread_join. A timed join (pthread_timedjoin_np) or making
    // Run() check exit_request frequently is better.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    int join_rc = pthread_join(unitree_thread_handle, NULL);
    if (join_rc != 0) {
      std::cerr << "Error joining Unitree SDK Bridge thread, rc: " << join_rc
                << ". It might be stuck." << std::endl;
    } else {
      std::cout << "Unitree SDK Bridge thread joined." << std::endl;
    }
  }

  std::cout << "Exiting main." << std::endl;
  return 0;
}
