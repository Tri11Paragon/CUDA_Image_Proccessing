#!/usr/bin/python3

# This script is used for initializing a CLion C++ Project

import requests
import json
import argparse
import os
import subprocess
import create_git_repo as repo
import util.color_io as io

scripts_dir = "/home/brett/Documents/code/scripts"
dir_path = os.path.dirname(os.path.realpath(__file__))
if not dir_path.endswith("/"):
	dir_path += "/"
github_url = "https://github.com/Tri11Paragon/"

git_ignore = """cmake-build*/
build/
out/
./cmake-build*/
./build/
./out/
"""

cmake_macros = """macro(sanitizers target_name)
    if (${ENABLE_ADDRSAN} MATCHES ON)
        target_compile_options(${target_name} PRIVATE -fsanitize=address)
        target_link_options(${target_name} PRIVATE -fsanitize=address)
    endif ()

    if (${ENABLE_UBSAN} MATCHES ON)
        target_compile_options(${target_name} PRIVATE -fsanitize=undefined)
        target_link_options(${target_name} PRIVATE -fsanitize=undefined)
    endif ()

    if (${ENABLE_TSAN} MATCHES ON)
        target_compile_options(${target_name} PRIVATE -fsanitize=thread)
        target_link_options(${target_name} PRIVATE -fsanitize=thread)
    endif ()
endmacro()

macro(compile_options target_name)
    if (NOT ${MOLD} STREQUAL MOLD-NOTFOUND)
        target_compile_options(${target_name} PUBLIC -fuse-ld=mold)
    endif ()

    target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
    target_link_options(${target_name} PRIVATE -Wall -Wextra -Wpedantic -Wno-comment)
    sanitizers(${target_name})
endmacro()

macro(blt_add_project name source type)

    project(${name}-${type})

    add_executable(${name}-${type} ${source})

    target_link_libraries(${name}-${type} PRIVATE BLT blt-gp Threads::Threads)

    compile_options(${name}-${type})
    target_compile_definitions(${name}-${type} PRIVATE BLT_DEBUG_LEVEL=${DEBUG_LEVEL})

    if (${TRACK_ALLOCATIONS})
        target_compile_definitions(${name}-${type} PRIVATE BLT_TRACK_ALLOCATIONS=1)
    endif ()

    add_test(NAME ${name} COMMAND ${name}-${type})

    set_property(TEST ${name} PROPERTY FAIL_REGULAR_EXPRESSION "FAIL;ERROR;FATAL;exception")

    project(${PROJECT_NAME})
endmacro()"""

cmake_exec_default_text = """cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

${MACROS}

project(${PROJECT_NAME} VERSION 0.0.1)

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)
option(BUILD_${PROJECT_NAME_UPPER}_EXAMPLES "Build example programs. This will build with CTest" OFF)
option(BUILD_${PROJECT_NAME_UPPER}_TESTS "Build test programs. This will build with CTest" OFF)

set(CMAKE_CXX_STANDARD ${CMAKE_LANGUAGE_VERSION})

${SUB_DIRS}

include_directories(include/)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(${PROJECT_NAME} ${PROJECT_BUILD_FILES})

compile_options(${PROJECT_NAME})

${LINKS}

if (${BUILD_${PROJECT_NAME_UPPER}_EXAMPLES})

endif()

if (BUILD_${PROJECT_NAME_UPPER}_TESTS)

endif()
"""

cmake_lib_default_text = """cmake_minimum_required(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})

${MACROS}

project(${PROJECT_NAME})

option(ENABLE_ADDRSAN "Enable the address sanitizer" OFF)
option(ENABLE_UBSAN "Enable the ub sanitizer" OFF)
option(ENABLE_TSAN "Enable the thread data race sanitizer" OFF)
option(BUILD_${PROJECT_NAME_UPPER}_EXAMPLES "Build example programs. This will build with CTest" OFF)
option(BUILD_${PROJECT_NAME_UPPER}_TESTS "Build test programs. This will build with CTest" OFF)

set(CMAKE_CXX_STANDARD ${CMAKE_LANGUAGE_VERSION})

${SUB_DIRS}

include_directories(include/)
file(GLOB_RECURSE PROJECT_BUILD_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_library(${PROJECT_NAME}${CMAKE_LIBRARY_TYPE} ${PROJECT_BUILD_FILES})

compile_options(${PROJECT_NAME})

${LINKS}

if (${BUILD_${PROJECT_NAME_UPPER}_EXAMPLES})

endif()

if (BUILD_${PROJECT_NAME_UPPER}_TESTS)

endif()
"""

default_main_file = """#include <iostream>

int main()
{
    std::cout << "Hello World!" << std::endl;    
}
"""

parser = argparse.ArgumentParser(prog='Bargo', description='Cargo but bad, for C++', epilog='Meow :3')
parser.add_argument("--cmake", default=None, help="Specify CMake version, defaults to using version installed on the system")
parser.add_argument("--cpp", "-p", default="17", help="C++ Version, defaults to C++17")
parser.add_argument("--no_git", default=True, action="store_false", help="Disables creating a git repo")
parser.add_argument("--create_git", "-c", nargs='?', const=True, default=False, help="Create the associated git repo", metavar="DESCRIPTION")
parser.add_argument("--no_blt", default=True, action="store_false", help="Disables init with BLT")
parser.add_argument("--lib", action="store_true", help="Create a lib instead of an exec")
parser.add_argument("--graphics", "-g", default=False, action="store_true", help="Init with graphics in mind, ie use BLT With Graphics")

parser.add_argument("action", help="Type of action to take, currently only 'init' is supported.")
parser.add_argument("name", help="Project Name")

args = parser.parse_args()

def open_process(command, print_out = True):
    process = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    str_out = stdout.decode('utf8')
    str_err = stderr.decode('utf8')
    if print_out and len(str_out) > 0:
        print(str_out, end='')
    if print_out and len(str_err) > 0:
        print(str_err, end='')
    #print(stdout, stderr, exit_code)
    return (stdout, stderr, exit_code)

def get_cmake_version():
    cmake_output, _, _ = open_process(["cmake", "--version"])

    cmake_lines = cmake_output.decode('utf8').splitlines()
    cmake_version = str(cmake_lines[0]).split("version ")[1].split('.')

    cmake_major = cmake_version[0]
    cmake_minor = cmake_version[1]
    
    return (cmake_major, cmake_minor)

def setup_dirs():
    open_process(["mkdir", "include", "src", "lib"])
    
def setup_blt(use_git, blt_url, blt_path):
    if use_git:
        open_process(["git", "submodule", "add", blt_url, "lib/" + blt_path])
        open_process(["git", "submodule", "update", "--remote", "--init", "--recursive"])
    else:
        open_process(["git", "clone", "--recursive", blt_url, "lib/" + blt_path])
    
def create_git_ignore():
    with open(".gitignore", "w") as f:
        f.write(git_ignore)

cmake_major, cmake_minor = get_cmake_version()
cpp_version = args.cpp
use_blt = args.no_blt
use_git = args.no_git
blt_url = "https://github.com/Tri11Paragon/BLT.git"
blt_path = "blt"
blt_lib = "BLT"
cmake_text = cmake_exec_default_text
project_name = args.name.replace(" ", "-").replace("_", "-")
sub_dirs = ""
links = ""

open_process(["mkdir", project_name])
wd = os.getcwd()
if not wd.endswith('/'):
    wd += "/"
wd += project_name
os.chdir(wd);

if args.lib:
    cmake_text = cmake_lib_default_text

if args.cmake:
    version_arr = args.cmake.split(".")
    if len(version_arr) < 2:
        io.red_print("Must provide at least major.minor CMake version!")
        exit()
    cmake_major = version_arr[0]
    cmake_minor = version_arr[1]

if args.graphics:
    blt_url = "https://git.tpgc.me/tri11paragon/BLT-With-Graphics-Template"
    blt_path = "blt-with-graphics"
    blt_lib = "BLT_WITH_GRAPHICS"
    
open_process(["cp", scripts_dir + "/commit.py", "./"]);

setup_dirs()

if use_git:
    open_process(["git", "init"])
    open_process(["git", "branch", "-M", "main"])
    create_git_ignore()

if use_blt:
    setup_blt(use_git=use_git, blt_url=blt_url, blt_path=blt_path)
    sub_dirs += "add_subdirectory(lib/" + blt_path + ")"
    links += "target_link_libraries(${PROJECT_NAME} PRIVATE " + blt_lib + ")"

if args.create_git:
    desc = ""
    if isinstance(args.create_git, str):
        desc = args.create_git
    open_process(["python3", dir_path + "create_git_repo.py", "-d", desc, project_name])
    if not github_url.endswith("/"):
        github_url += "/"
    open_process(["git", "remote", "add", "origin", github_url + project_name])
    open_process(["git", "branch", "--set-upstream-to=origin/main", "main"])
    
cmake_text = cmake_text.replace("${MACROS}", cmake_macros)
cmake_text = cmake_text.replace("${SUB_DIRS}", sub_dirs)
cmake_text = cmake_text.replace("${LINKS}", links)
cmake_text = cmake_text.replace("${CMAKE_MAJOR_VERSION}", cmake_major)
cmake_text = cmake_text.replace("${CMAKE_MINOR_VERSION}", cmake_minor)
cmake_text = cmake_text.replace("${PROJECT_NAME}", project_name)
cmake_text = cmake_text.replace("${PROJECT_NAME_UPPER}", project_name.upper().replace("-", "_"))
cmake_text = cmake_text.replace("${CMAKE_LANGUAGE_VERSION}", cpp_version)

with open("CMakeLists.txt", "w") as f:
    f.write(cmake_text)
        
with open("src/main.cpp", "w") as f:
    f.write(default_main_file)

print("Created " + project_name + "!")

