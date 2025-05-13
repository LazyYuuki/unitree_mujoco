cmake_minimum_required(VERSION 3.16)
project(unitree_mujoco)
                                                 
enable_language(C)
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# Fetch and build MuJoCo from source
FetchContent_Declare(
    mujoco
    GIT_REPOSITORY https://github.com/menloresearch/mujoco-wasm.git
    GIT_TAG main
)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE BOOL "Set CMAKE policy version" FORCE)
FetchContent_MakeAvailable(mujoco)

FetchContent_Declare(
    unitree_sdk2
    GIT_REPOSITORY https://github.com/unitreerobotics/unitree_sdk2.git
    GIT_TAG main
)
set(BUILD_EXAMPLES OFF CACHE BOOL "Ignore examples" FORCE)
FetchContent_MakeAvailable(unitree_sdk2)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG yaml-cpp-0.7.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Don't build yaml-cpp tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Don't build yaml-cpp tools" FORCE)
FetchContent_MakeAvailable(yaml-cpp)
   
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics/lib/cmake")

file(GLOB SIM_SRC
    src/joystick/joystick.cc
    src/unitree_sdk2_bridge/*.cc)

set(SIM_DEPENDENCIES
    pthread
    mujoco::mujoco
    yaml-cpp
    unitree_sdk2)

add_executable(unitree_mujoco ${SIM_SRC} src/main.cc)
target_link_libraries(unitree_mujoco ${SIM_DEPENDENCIES})

add_executable(test test/test_unitree_sdk2.cpp)
target_link_libraries(test unitree_sdk2)

add_executable(jstest src/joystick/jstest.cc src/joystick/joystick.cc)

set(CMAKE_BUILD_TYPE Release)
