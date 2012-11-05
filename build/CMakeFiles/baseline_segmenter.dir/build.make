# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build

# Include any dependencies generated for this target.
include CMakeFiles/baseline_segmenter.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/baseline_segmenter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/baseline_segmenter.dir/flags.make

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: CMakeFiles/baseline_segmenter.dir/flags.make
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: ../src/baseline_segmenter.cpp
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: ../manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/external/opencv_bleeding/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/external/opencv2_flags/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/eigen_flags/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/gzstream/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/timer/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/eigen_extensions/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/serializable/manifest.xml
CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/core/image_labeler/manifest.xml
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -o CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o -c /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/baseline_segmenter.cpp

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -E /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/baseline_segmenter.cpp > CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.i

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -S /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/baseline_segmenter.cpp -o CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.s

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.requires:
.PHONY : CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.requires

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.provides: CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.requires
	$(MAKE) -f CMakeFiles/baseline_segmenter.dir/build.make CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.provides.build
.PHONY : CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.provides

CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.provides.build: CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o

# Object files for target baseline_segmenter
baseline_segmenter_OBJECTS = \
"CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o"

# External object files for target baseline_segmenter
baseline_segmenter_EXTERNAL_OBJECTS =

../bin/baseline_segmenter: CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o
../bin/baseline_segmenter: ../lib/libsegmentation_and_tracking.so
../bin/baseline_segmenter: CMakeFiles/baseline_segmenter.dir/build.make
../bin/baseline_segmenter: CMakeFiles/baseline_segmenter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/baseline_segmenter"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/baseline_segmenter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/baseline_segmenter.dir/build: ../bin/baseline_segmenter
.PHONY : CMakeFiles/baseline_segmenter.dir/build

CMakeFiles/baseline_segmenter.dir/requires: CMakeFiles/baseline_segmenter.dir/src/baseline_segmenter.o.requires
.PHONY : CMakeFiles/baseline_segmenter.dir/requires

CMakeFiles/baseline_segmenter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/baseline_segmenter.dir/cmake_clean.cmake
.PHONY : CMakeFiles/baseline_segmenter.dir/clean

CMakeFiles/baseline_segmenter.dir/depend:
	cd /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build/CMakeFiles/baseline_segmenter.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/baseline_segmenter.dir/depend
