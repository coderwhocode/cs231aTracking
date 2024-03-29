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
include CMakeFiles/hand_segmenter.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/hand_segmenter.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hand_segmenter.dir/flags.make

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: CMakeFiles/hand_segmenter.dir/flags.make
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: ../src/hand_segmenter.cpp
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: ../manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/external/opencv_bleeding/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/external/opencv2_flags/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/eigen_flags/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/gzstream/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/timer/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/eigen_extensions/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/utility/serializable/manifest.xml
CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o: /home/sandra/cs231a/jarvis/ros-pkg/core/image_labeler/manifest.xml
	$(CMAKE_COMMAND) -E cmake_progress_report /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -o CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o -c /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/hand_segmenter.cpp

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/hand_segmenter.dir/src/hand_segmenter.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -E /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/hand_segmenter.cpp > CMakeFiles/hand_segmenter.dir/src/hand_segmenter.i

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/hand_segmenter.dir/src/hand_segmenter.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -W -Wall -Wno-unused-parameter -fno-strict-aliasing -pthread -S /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/src/hand_segmenter.cpp -o CMakeFiles/hand_segmenter.dir/src/hand_segmenter.s

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.requires:
.PHONY : CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.requires

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.provides: CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.requires
	$(MAKE) -f CMakeFiles/hand_segmenter.dir/build.make CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.provides.build
.PHONY : CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.provides

CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.provides.build: CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o

# Object files for target hand_segmenter
hand_segmenter_OBJECTS = \
"CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o"

# External object files for target hand_segmenter
hand_segmenter_EXTERNAL_OBJECTS =

../bin/hand_segmenter: CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o
../bin/hand_segmenter: ../lib/libsegmentation_and_tracking.so
../bin/hand_segmenter: CMakeFiles/hand_segmenter.dir/build.make
../bin/hand_segmenter: CMakeFiles/hand_segmenter.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable ../bin/hand_segmenter"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hand_segmenter.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hand_segmenter.dir/build: ../bin/hand_segmenter
.PHONY : CMakeFiles/hand_segmenter.dir/build

CMakeFiles/hand_segmenter.dir/requires: CMakeFiles/hand_segmenter.dir/src/hand_segmenter.o.requires
.PHONY : CMakeFiles/hand_segmenter.dir/requires

CMakeFiles/hand_segmenter.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hand_segmenter.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hand_segmenter.dir/clean

CMakeFiles/hand_segmenter.dir/depend:
	cd /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build /home/sandra/cs231a/jarvis/ros-pkg/core/segmentation_and_tracking/build/CMakeFiles/hand_segmenter.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/hand_segmenter.dir/depend

