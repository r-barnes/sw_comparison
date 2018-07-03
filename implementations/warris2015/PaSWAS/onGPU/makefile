-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(CU_DEPS)),)
-include $(CU_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
endif

CUDA_HOME=/opt/nvidia/cudatoolkit9.1/9.1.85_3.10-1.0502.df1cc54.3.1/

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: paswas

# Tool invocations
paswas: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C Linker'
	gcc -L${CUDA_HOME}/lib64 -o"paswas" $(OBJS) $(USER_OBJS) $(LIBS) -lstdc++
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(CU_DEPS)$(OBJS)$(C_DEPS)$(EXECUTABLES) paswas gpuAlign* smithwaterman*
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
