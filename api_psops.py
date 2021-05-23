from ctypes import *
import platform
import os
from sys import float_repr_style, stderr
import numpy as np
import datetime
from numba import jitclass
from numpy.core.fromnumeric import nonzero, size
from numpy.core.multiarray import inner
from numpy.lib.function_base import select

parent_dir, _ = os.path.split(os.path.abspath(__file__))

array_1d_double = np.ctypeslib.ndpointer(
    dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_int = np.ctypeslib.ndpointer(dtype=np.int, ndim=1, flags='CONTIGUOUS')

if platform.system() == 'Windows':
    os.environ['path'] += ';%s\\dll_win' % parent_dir
elif platform.system() == 'Linux':
    os.environ['PATH'] += ';%s\\dll_linux' % parent_dir
else:
    print('Unknown operating system. Please check!')
    exit(-1)


class psopsAPI:
    ################################################################################################
    # construct
    ################################################################################################
    def __init__(self, flg=0, rng=None):
        # api flag
        self.__flg = flg
        # random state
        self.set_random_state(np.random.default_rng() if rng is None else rng)
        # working direction
        self.__workingDir = parent_dir
        # load dll
        dll_path = self.__workingDir
        # load dll
        if platform.system() == 'Windows':
            dll_path += '/dll_win/PSOPS-Console-QT-V.dll'
        elif platform.system() == 'Linux':
            dll_path += '/dll_linux/libPSOPS-Console-QT-V.so.1.0.0'
        else:
            print('Unknown operating system. Please check!')
            exit(-1)
        self._load_dll(dll_path)
        # load config file
        self._load_configuration(self.__workingDir + '/config.txt')
        # psops function test
        self._basic_fun_test()
        # basic info
        self._get_basic_info()
        # create buffer
        self._create_buffer()
        # get initial state
        self._get_initial_state()
        # time stamp
        self.__timeStamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
        # current total_step
        self.__cur_total_step = -1
        # bounds
        self.__generator_p_bounds = [0.5, 1.5]
        self.__load_pq_bounds = [0.7, 1.2]
        print('api for psops creation successful.')

    # deconstruct
    def __del__(self):
        print('api for psops deletion successful.')

    # dll api with PSOPS
    def _load_dll(self, dll_path):
        self.__psDLL = cdll.LoadLibrary(dll_path)
        # MPI, currently useless
        self.__psDLL.init_MPI.argtypes = None
        self.__psDLL.init_MPI.restype = c_bool
        self.__psDLL.mapping_MPI.argtypes = None
        self.__psDLL.mapping_MPI.restype = c_bool
        self.__psDLL.finalize_MPI.argtypes = None
        self.__psDLL.finalize_MPI.restype = c_bool
        # basic fun
        self.__psDLL.read_Settings.argtypes = [c_wchar_p, c_int]
        self.__psDLL.read_Settings.restype = c_bool
        self.__psDLL.cal_Functions.argtypes = [c_int]
        self.__psDLL.cal_Functions.restype = c_bool
        # calculation
        self.__psDLL.cal_PF_Basic_Power_Flow.argtypes = None
        self.__psDLL.cal_PF_Basic_Power_Flow.restype = c_bool
        self.__psDLL.cal_TS_Simulation_TI_SV.argtypes = [c_double, c_int]
        self.__psDLL.cal_TS_Simulation_TI_SV.restype = c_int
        # cal info and cal control
        self.__psDLL.get_TE.argtypes = None
        self.__psDLL.get_TE.restype = c_double
        self.__psDLL.get_DT.argtypes = None
        self.__psDLL.get_DT.restype = c_double
        self.__psDLL.get_Max_Step.argtypes = None
        self.__psDLL.get_Max_Step.restype = c_int
        self.__psDLL.get_Finish_Step.argtypes = None
        self.__psDLL.get_Finish_Step.restype = c_int
        self.__psDLL.get_Fault_Step_Sequence.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Fault_Step_Sequence.restype = c_bool
        self.__psDLL.set_TS_Step_Network_State.argtypes = [c_int, c_int]
        self.__psDLL.set_TS_Step_Network_State.restype = c_bool
        self.__psDLL.set_TS_Step_Element_State.argtypes = [c_int, c_int]
        self.__psDLL.set_TS_Step_Element_State.restype = c_bool
        self.__psDLL.set_TS_Step_All_State.argtypes = [c_int, c_int]
        self.__psDLL.set_TS_Step_All_State.restype = c_bool
        # asynchronous systems
        self.__psDLL.get_N_ACSystem.argtypes = None
        self.__psDLL.get_N_ACSystem.restype = c_int
        self.__psDLL.get_ACSystem_TS_CurStep_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACSystem_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_ACSystem_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACSystem_TS_All_Result.restype = c_bool
        # bus
        self.__psDLL.get_N_Bus.argtypes = [c_int]
        self.__psDLL.get_N_Bus.restype = c_int
        self.__psDLL.get_Bus_Name.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_Name.restype = c_char_p
        self.__psDLL.get_Bus_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Bus_Sys_No.restype = c_int
        self.__psDLL.get_Bus_VMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_VMax.restype = c_double
        self.__psDLL.get_Bus_VMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Bus_VMin.restype = c_double
        self.__psDLL.get_Bus_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Bus_LF_Result.restype = c_bool
        self.__psDLL.get_Bus_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Bus_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Bus_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Bus_TS_All_Result.restype = c_bool
        # ac line
        self.__psDLL.get_N_ACLine.argtypes = [c_int]
        self.__psDLL.get_N_ACLine.restype = c_int
        self.__psDLL.get_ACLine_I_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_I_No.restype = c_int
        self.__psDLL.get_ACLine_J_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_J_No.restype = c_int
        self.__psDLL.get_ACLine_Sys_No.argtypes = [c_int]
        self.__psDLL.get_ACLine_Sys_No.restype = c_int
        self.__psDLL.get_ACLine_No.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_No.restype = c_long
        self.__psDLL.get_ACLine_Current_Capacity.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_Current_Capacity.restype = c_double
        self.__psDLL.get_ACLine_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_ACLine_LF_Result.restype = c_bool
        self.__psDLL.get_ACLine_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_ACLine_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_ACLine_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_ACLine_TS_All_Result.restype = c_bool
        # transformer
        self.__psDLL.get_N_Transformer.argtypes = [c_int]
        self.__psDLL.get_N_Transformer.restype = c_int
        self.__psDLL.get_Transformer_I_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_I_No.restype = c_int
        self.__psDLL.get_Transformer_J_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_J_No.restype = c_int
        self.__psDLL.get_Transformer_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Transformer_Sys_No.restype = c_int
        self.__psDLL.get_Transformer_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_No.restype = c_long
        self.__psDLL.get_Transformer_Current_Capacity.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_Current_Capacity.restype = c_double
        self.__psDLL.get_Transformer_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Transformer_LF_Result.restype = c_bool
        self.__psDLL.get_Transformer_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Transformer_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Transformer_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Transformer_TS_All_Result.restype = c_bool
        # generator
        self.__psDLL.get_N_Generator.argtypes = [c_int]
        self.__psDLL.get_N_Generator.restype = c_int
        self.__psDLL.get_Generator_Bus_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Bus_No.restype = c_int
        self.__psDLL.get_Generator_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Generator_Sys_No.restype = c_int
        self.__psDLL.get_Generator_LF_Bus_Type.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_LF_Bus_Type.restype = c_int
        self.__psDLL.get_Generator_V0.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_V0.restype = c_double
        self.__psDLL.set_Generator_V0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Generator_V0.restype = c_bool
        self.__psDLL.get_Generator_P0.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_P0.restype = c_double
        self.__psDLL.set_Generator_P0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Generator_P0.restype = c_bool
        self.__psDLL.get_Generator_PMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PMax.restype = c_double
        self.__psDLL.get_Generator_PMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PMin.restype = c_double
        self.__psDLL.get_Generator_QMax.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_QMax.restype = c_double
        self.__psDLL.get_Generator_QMin.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_QMin.restype = c_double
        self.__psDLL.get_Generator_Tj.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Tj.restype = c_double
        self.__psDLL.get_Generator_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_LF_Result.restype = c_bool
        self.__psDLL.get_Generator_TS_Result_Dimension.argtypes = [c_int, c_int, c_bool]
        self.__psDLL.get_Generator_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int, c_bool]
        self.__psDLL.get_Generator_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Generator_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Generator_TS_All_Result.restype = c_bool
        # exiter
        self.__psDLL.get_Generator_Exciter_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Exciter_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_Exciter_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_Exciter_TS_CurStep_Result.restype = c_bool
        # governer
        self.__psDLL.get_Generator_Governor_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Governor_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_Governor_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_Governor_TS_CurStep_Result.restype = c_bool
        # pss
        self.__psDLL.get_Generator_PSS_TS_Result_Dimension.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_PSS_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Generator_PSS_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Generator_PSS_TS_CurStep_Result.restype = c_bool
        # load
        self.__psDLL.get_N_Load.argtypes = [c_int]
        self.__psDLL.get_N_Load.restype = c_int
        self.__psDLL.get_Load_Bus_No.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_Bus_No.restype = c_int
        self.__psDLL.get_Load_Sys_No.argtypes = [c_int]
        self.__psDLL.get_Load_Sys_No.restype = c_int
        self.__psDLL.get_Load_P0.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_P0.restype = c_double
        self.__psDLL.set_Load_P0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Load_P0.restype = c_bool
        self.__psDLL.get_Load_Q0.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_Q0.restype = c_double
        self.__psDLL.set_Load_Q0.argtypes = [c_double, c_int, c_int]
        self.__psDLL.set_Load_Q0.restype = c_bool
        self.__psDLL.get_Load_LF_Result.argtypes = [array_1d_double, c_int, c_int]
        self.__psDLL.get_Load_LF_Result.restype = c_bool
        self.__psDLL.get_Load_TS_Result_Dimension.argtypes = [c_int, c_int, c_bool]
        self.__psDLL.get_Load_TS_Result_Dimension.restype = c_int
        self.__psDLL.get_Load_TS_CurStep_Result.argtypes = [array_1d_double, c_int, c_int, c_bool]
        self.__psDLL.get_Load_TS_CurStep_Result.restype = c_bool
        self.__psDLL.get_Load_TS_All_Result.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Load_TS_All_Result.restype = c_bool
        # network
        self.__psDLL.get_N_Non_Zero_Element.argtypes = [c_int]
        self.__psDLL.get_N_Non_Zero_Element.restype = c_int
        self.__psDLL.get_N_Inverse_Non_Zero_Element.argtypes = [c_int]
        self.__psDLL.get_N_Inverse_Non_Zero_Element.restype = c_int
        self.__psDLL.get_N_ACSystem_Check_Connectivity.argtypes = [c_int]
        self.__psDLL.get_N_ACSystem_Check_Connectivity.restype = c_int
        self.__psDLL.get_ACLine_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_ACLine_Connectivity.restype = c_bool
        self.__psDLL.set_ACLine_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_ACLine_Connectivity.restype = c_bool
        self.__psDLL.get_Transformer_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Transformer_Connectivity.restype = c_bool
        self.__psDLL.set_Transformer_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Transformer_Connectivity.restype = c_bool
        self.__psDLL.set_Rebuild_All_Network_Data.argtypes = None
        self.__psDLL.set_Rebuild_All_Network_Data.restype = c_bool
        self.__psDLL.get_Generator_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Generator_Connectivity.restype = c_bool
        self.__psDLL.set_Generator_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Generator_Connectivity.restype = c_bool
        self.__psDLL.get_Load_Connectivity.argtypes = [c_int, c_int]
        self.__psDLL.get_Load_Connectivity.restype = c_bool
        self.__psDLL.set_Load_Connectivity.argtypes = [c_bool, c_int, c_int]
        self.__psDLL.set_Load_Connectivity.restype = c_bool
        self.__psDLL.get_Admittance_Matrix_Full.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Admittance_Matrix_Full.restype = c_bool
        self.__psDLL.get_Impedence_Matrix_Full.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Impedence_Matrix_Full.restype = c_bool
        self.__psDLL.get_Impedence_Matrix_Factorized.argtypes = [array_1d_double, c_int]
        self.__psDLL.get_Impedence_Matrix_Factorized.restype = c_bool
        # fault and disturbance
        self.__psDLL.set_Fault_Disturbance_Clear_All.argtypes = None
        self.__psDLL.set_Fault_Disturbance_Clear_All.restype = c_bool
        self.__psDLL.set_Fault_Disturbance_Add_Fault.argtypes = [c_int, c_double, c_double, c_double, c_int, c_int, c_bool, c_int]
        self.__psDLL.set_Fault_Disturbance_Add_Fault.restype = c_bool
        self.__psDLL.set_Fault_Disturbance_Add_Disturbance.argtypes = [c_int, c_double, c_double, c_int, c_int, c_bool, c_int]
        self.__psDLL.set_Fault_Disturbance_Add_Disturbance.restype = c_bool

    # loading configuration file
    def _load_configuration(self, cfg_path):
        self.__config_path = cfg_path
        cfg = open(cfg_path, "r")
        for line in cfg.readlines():
            if line[0:3].lower() == 'dir':
                self.__fullFilePath = self.__workingDir + line[4:].strip()[1:]
                (self.__absFilePath, temp_file_name) = os.path.split(self.__fullFilePath)
                (self.__absFileName, extension) = os.path.splitext(temp_file_name)
                print(self.__absFilePath, self.__absFileName, extension)
        cfg.close()

    # basci function test before utilization
    def _basic_fun_test(self):
        assert self.__psDLL.read_Settings(self.__config_path, len(self.__config_path)), 'read settings failure!'
        assert self.__psDLL.cal_Functions(1), 'basic function check failure!'

    # get basci information of the power system
    def _get_basic_info(self):
        self.__nACSystem = self.__psDLL.get_N_ACSystem()
        self.__nBus = self.__psDLL.get_N_Bus(-1)
        assert self.__nBus >= 0, 'system total bus number wrong, please check!'
        self.__allBusName = list()
        for bus_no in range(self.__nBus):
            tmp = self.__psDLL.get_Bus_Name(bus_no, -1)
            assert tmp is not None,  "bus name is empty, please check bus no.!"
            self.__allBusName.append(string_at(tmp, -1).decode('gbk'))
        self.__allBusName = np.array(self.__allBusName)
        self.__nACLine = self.__psDLL.get_N_ACLine(-1)
        assert self.__nACLine >= 0, 'system number wrong, please check!'
        self.__nTransformer = self.__psDLL.get_N_Transformer(-1)
        assert self.__nTransformer >= 0, 'total number of transformer wrong, please check!'
        self.__nGenerator = self.__psDLL.get_N_Generator(-1)
        assert self.__nGenerator >= 0, 'total number of generator wrong, please check!'
        self.__nLoad = self.__psDLL.get_N_Load(-1)
        assert self.__nLoad >= 0, 'total number of load wrong, please check!'
        self.__nNonzero = self.__psDLL.get_N_Non_Zero_Element(-1)
        assert self.__nNonzero >= 0, 'total number of non-zero element wrong, please check!'
        self.__nInverseNonZero = self.__psDLL.get_N_Inverse_Non_Zero_Element(-1)
        assert self.__nInverseNonZero >= 0, 'total number of inverse non-zeror wrong, please check!'

    # create buffer
    def _create_buffer(self):
        self.__bufferSize = max(
            self.__nBus * 6 * 2000, max(self.__nNonzero, self.__nInverseNonZero) * 6)
        self.__doubleBuffer = np.zeros(self.__bufferSize, np.float64)
        self.__intBuffer = np.zeros(self.__bufferSize, np.int)
        self.__boolBuffer = np.zeros(self.__bufferSize, np.bool)
        print("Buffer Created in Python.")
        # run buffer tests
        # self._buffer_tests()

    def _get_initial_state(self):
        # gen lf bus type
        bus_type = self.get_generator_all_lf_bus_type()
        self.__indexSlack = np.arange(self.__nGenerator, dtype=np.int)[bus_type == 'slack']
        self.__indexCtrlGen = np.arange(self.__nGenerator, dtype=np.int)[bus_type != 'slack']
        # gen v set
        self.__generator_v_origin = self.get_generator_all_v_set()
        # gen p set
        self.__generator_p_origin = self.get_generator_all_p_set()
        # load p set
        self.__load_p_origin = self.get_load_all_p_set()
        # load q set
        self.__load_q_origin = self.get_load_all_q_set()

    ################################################################################################
    # calculation
    ################################################################################################
    # power flow, basic power flow calculation, N-R
    def cal_pf_basic_power_flow_nr(self):
        return self.__psDLL.cal_PF_Basic_Power_Flow()

    # transient stability, ti, sparse vector
    def cal_ts_simulation_ti_sv(self, start_time=0.0, contingency_no=0):
        return self.__psDLL.cal_TS_Simulation_TI_SV(start_time, contingency_no)

    ################################################################################################
    # cal info and cal control
    ################################################################################################
    # transient stability, t end
    def get_info_ts_end_t(self):
        return self.__psDLL.get_TE()

    # transient stability, delata t
    def get_info_ts_delta_t(self):
        return self.__psDLL.get_DT()

    # transient stability, max step
    def get_info_ts_max_step(self):
        return self.__psDLL.get_Max_Step()

    # transient stability, finish step
    def get_info_ts_finish_step(self):
        self.__cur_total_step = self.__psDLL.get_Finish_Step()
        return self.__cur_total_step

    # fault step sequence, [0:1] is n_fault_step, [1:] is step sequences
    def get_fault_step_sequence(self, contingency_no=0):
        assert self.__psDLL.get_Fault_Step_Sequence(self.__doubleBuffer, contingency_no) is True, "get fault time sequence failed, please check!"
        n_fault_step = int(self.__doubleBuffer[0])
        return self.__doubleBuffer[1:1+n_fault_step].astype(np.int32)

    # transient stability, step network status
    def set_info_ts_step_network_state(self, real_step, sys_no=-1):
        assert self.__psDLL.set_TS_Step_Network_State(real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    # transient stability, step element status
    def set_info_ts_step_element_state(self, real_step, sys_no=-1):
        assert self.__psDLL.set_TS_Step_Element_State(real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    # transient stability, step all status
    def set_info_ts_step_all_state(self, real_step, sys_no=-1):
        assert self.__psDLL.set_TS_Step_All_State(real_step, sys_no) is True, "Step No or AC System No error. Please check!"

    ################################################################################################
    # ac systems
    ################################################################################################
    def get_acsystem_number(self):
        return self.__nACSystem

    # transient stability, current step, system variable, result, time, max delta, min freq, max freq, min vol, max vol
    def get_acsystem_ts_cur_step_result(self, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACSystem_TS_CurStep_Result(buffer, sys_no) == True, "get ts system variable result failed, please check!"
        if rt is True:
            n_system = self.__nACSystem if sys_no == -1 else 1
            return buffer[:n_system*6].astype(np.float32)

    # transient stability, step, system variable, result, time, max delta, min freq, max freq, min vol, max vol
    def get_acsystem_ts_step_result(self, step, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_acsystem_ts_cur_step_result(sys_no)

    # transient stability, all steps, all system variable, result, time, max delta, min freq, max freq, min vol, max vol
    def get_acsystem_all_ts_result(self, sys_no=-1):
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_ACSystem_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all system variable results failed!"
        n_system = self.__nACSystem if sys_no == -1 else 1
        return self.__doubleBuffer[0:n_system * total_step * 6].reshape(n_system, total_step, 6).astype(np.float32)

    ################################################################################################
    # buses
    ################################################################################################
    # total bus number
    def get_bus_number(self, sys_no=-1):
        n_bus = self.__nBus if sys_no == -1 else self.__psDLL.get_N_Bus(sys_no)
        assert n_bus >= 0, 'total number of bus wrong, please check!'
        return n_bus

    # bus name
    def get_bus_name(self, bus_no, sys_no=-1):
        if sys_no == -1:
            return self.__allBusName[bus_no]
        bus_name = self.__psDLL.get_Bus_Name(bus_no, sys_no)
        assert bus_name is not None,  "bus name is empty, please check sys/bus no!"
        return string_at(bus_name, -1).decode('gbk')

    # all bus name
    def get_bus_all_name(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        if sys_no == -1:
            return self.__allBusName[bus_list]
        bus_names = list()
        for bus_no in bus_list:
            bus_names.append(self.get_bus_name(bus_no, sys_no))
        return np.array(bus_names)

    # bus No.
    def get_bus_no(self, name):
        bus_no = np.where(self.__allBusName == name)[0]
        assert len(bus_no) == 1, "bus name duplication"
        return bus_no[0]

    # asynchronous system No.
    def get_bus_sys_no(self, bus_no):
        sys_no = self.__psDLL.get_Bus_Sys_No(bus_no)
        assert sys_no >= 0, "bus asynchronous system detection failed!"
        return sys_no

    # asynchronous system No. of all buses
    def get_bus_all_sys_no(self, bus_list=None):
        bus_list = np.arange(self.__nBus, dtype=np.int) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__intBuffer[index] = self.get_bus_sys_no(bus_no)
        return self.__intBuffer[:len(bus_list)].astype(np.int32)

    # bus vmax
    def get_bus_vmax(self, bus_no, sys_no=-1):
        vmax = self.__psDLL.get_Bus_VMax(bus_no, sys_no)
        assert vmax > -1.0e10, "vmax wrong, please check!"
        return vmax

    # all bus vmax
    def get_bus_all_vmax(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__doubleBuffer[index] = self.get_bus_vmax(bus_no, sys_no)
        return self.__doubleBuffer[:len(bus_list)].astype(np.float32)

    # bus vmin
    def get_bus_vmin(self, bus_no, sys_no=-1):
        vmin = self.__psDLL.get_Bus_VMin(bus_no, sys_no)
        assert vmin > -1.0e10, "vmin wrong, please check!"
        return vmin

    # all bus vmin
    def get_bus_all_vmin(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.__doubleBuffer[index] = self.get_bus_vmin(bus_no, sys_no)
        return self.__doubleBuffer[:len(bus_list)].astype(np.float32)

    # load flow bus result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_lf_result(self, bus_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Bus_LF_Result(buffer, bus_no, sys_no) == True, "get lf bus result failed, please check!"
        if rt is True:
            return buffer[:6].astype(np.float32)

    # load flow all bus result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_all_lf_result(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.get_bus_lf_result(bus_no, sys_no, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(bus_list)*6].reshape(len(bus_list), 6).astype(np.float32)

    # transient stability, current step, bus, result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_ts_cur_step_result(self, bus_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Bus_TS_CurStep_Result(buffer, bus_no, sys_no) == True, "get ts bus result failed, please check!"
        if rt is True:
            return buffer[:6].astype(np.float32)

    # transient stability, current step, all bus, result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_all_ts_cur_step_result(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        for (index, bus_no) in zip(range(len(bus_list)), bus_list):
            self.get_bus_ts_cur_step_result(bus_no, sys_no, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(bus_list)*6].reshape(len(bus_list), 6).astype(np.float32)

    # transient stability, step, bus, result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_ts_step_result(self, step, bus_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_bus_ts_cur_step_result(bus_no, sys_no)

    # transient stability, current step, all bus, result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_all_ts_step_result(self, step, bus_list=None, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_bus_all_ts_cur_step_result(bus_list, sys_no)

    # transient stability, all steps, all bus, result, V, θ, Pg, Qg, Pl, Ql
    def get_bus_all_ts_result(self, bus_list=None, sys_no=-1):
        bus_list = np.arange(self.get_bus_number(sys_no), dtype=np.int) if bus_list is None else bus_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Bus_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all bus results failed!"
        all_result = self.__doubleBuffer[0:total_step * 6 * self.get_bus_number(sys_no)].reshape(total_step, self.get_bus_number(sys_no), 6)
        return all_result[:, bus_list, :].astype(np.float32)

    ################################################################################################
    # aclines
    ################################################################################################
    # total number of ac line
    def get_acline_number(self, sys_no=-1):
        n_acline = self.__nACLine if sys_no == -1 else self.__psDLL.get_N_ACLine(sys_no)
        assert n_acline >= 0, 'total number of ac line wrong, please check!'
        return n_acline

    # acline i no
    def get_acline_i_no(self, lnew, sys_no=-1):
        i_no = self.__psDLL.get_ACLine_I_No(lnew, sys_no)
        assert i_no >= 0, "ac line i no wrong, please check!"
        return i_no

    # all acline i no
    def get_acline_all_i_no(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_i_no(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    # acline j no
    def get_acline_j_no(self, lnew, sys_no=-1):
        j_no = self.__psDLL.get_ACLine_J_No(lnew, sys_no)
        assert j_no >= 0, "ac line j no, please check!"
        return j_no

    # all acline j no
    def get_acline_all_j_no(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_j_no(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    # acline i name
    def get_acline_i_name(self, lnew, sys_no=-1):
        return self.get_bus_name(self.get_acline_i_no(lnew, sys_no), sys_no)

    # all acline i name
    def get_acline_all_i_name(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        return self.get_bus_all_name(None, sys_no)[self.get_acline_all_i_no(acline_list)]

    # acline j name
    def get_acline_j_name(self, lnew, sys_no=-1):
        return self.get_bus_name(self.get_acline_j_no(lnew, sys_no), sys_no)

    # all acline j name
    def get_acline_all_j_name(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        return self.get_bus_all_name(None, sys_no)[self.get_acline_all_j_no(acline_list)]

    # acline asynchronous system no
    def get_acline_sys_no(self, line_no):
        sys_no = self.__psDLL.get_ACLine_Sys_No(line_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    # all acline asynchronous system no
    def get_acline_all_sys_no(self, acline_list=None):
        acline_list = np.arange(self.get_acline_number(), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_sys_no(acline_no)
        return self.__intBuffer[:len(acline_list)].astype(np.int32)

    # acline NO
    def get_acline_NO(self, acline_no, sys_no=-1):
        a_NO = self.__psDLL.get_ACLine_No(acline_no, sys_no)
        assert a_NO >= 0, "ac line No is wrong, please check!"
        return a_NO

    # all acline NO
    def get_acline_all_NO(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__intBuffer[index] = self.get_acline_NO(acline_no, sys_no)
        return self.__intBuffer[:len(acline_list)]

    # acline current capacity
    def get_acline_current_capacity(self, acline_no, sys_no=-1):
        current_capacity = self.__psDLL.get_ACLine_Current_Capacity(acline_no, sys_no)
        assert current_capacity > 0.0, "ac line current capacity wrong, please check!"
        return current_capacity

    # acline all current capacity
    def get_acline_all_current_capacity(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.__doubleBuffer[index] = self.get_acline_current_capacity(acline_no, sys_no)
        return self.__doubleBuffer[:len(acline_list)].astype(np.float32)

    # load flow acline result, pi, qi, pj, qj
    def get_acline_lf_result(self, acline_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACLine_LF_Result(buffer, acline_no, sys_no) == True, "get lf ac line result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    # load flow all acline result, pi, qi, pj, qj
    def get_acline_all_lf_result(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.get_acline_lf_result(acline_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(acline_list)*4].reshape(len(acline_list), 4).astype(np.float32)

    # transient stability, current step, acline, result, pi, qi, pj, qj
    def get_acline_ts_cur_step_result(self, acline_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_ACLine_TS_CurStep_Result(buffer, acline_no, sys_no) == True, "get ts acline result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    # transient stability, current step, all acline, result, pi, qi, pj, qj
    def get_acline_all_ts_cur_step_result(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)), acline_list):
            self.get_acline_ts_cur_step_result(acline_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(acline_list)*4].reshape(len(acline_list), 4).astype(np.float32)

    # transient stability, step, acline, result, pi, qi, pj, qj
    def get_acline_ts_step_result(self, step, acline_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_acline_ts_cur_step_result(acline_no, sys_no)

    # transient stability, current step, all acline, result, pi, qi, pj, qj
    def get_acline_all_ts_step_result(self, step, acline_list=None, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_acline_all_ts_cur_step_result(acline_list, sys_no)

    # transient stability, all steps, all acline, result, pi, qi, pj, qj
    def get_acline_all_ts_result(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_ACLine_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all acline results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_acline_number(sys_no)].reshape(total_step, self.get_acline_number(sys_no), 4)
        return all_result[:, acline_list, :].astype(np.float32)

    # acline info
    def get_acline_info(self, lnew, sys_no=-1):
        return [self.get_acline_i_name(lnew, sys_no), self.get_acline_j_name(lnew, sys_no), self.get_acline_NO(lnew, sys_no)]
    
    # acline all info
    def get_acline_all_info(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        return [self.get_acline_all_i_name(acline_list, sys_no), self.get_acline_all_j_name(acline_list, sys_no), self.get_acline_all_NO(acline_list, sys_no)]

    ################################################################################################
    # transformers
    ################################################################################################
    # total number of transformer
    def get_transformer_number(self, sys_no=-1):
        if sys_no == -1:
            return self.__nTransformer
        n_transformer = self.__psDLL.get_N_Transformer(sys_no)
        assert n_transformer >= 0, 'total number of transformer wrong, please check!'
        return n_transformer

    # transformer i no
    def get_transformer_i_no(self, tnew, sys_no=-1):
        i_no = self.__psDLL.get_Transformer_I_No(tnew, sys_no)
        assert i_no >= 0, "transformer i no wrong, please check!"
        return i_no

    # all transformer i no
    def get_transformer_all_i_no(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_i_no(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    # transformer j no
    def get_transformer_j_no(self, tnew, sys_no=-1):
        j_no = self.__psDLL.get_Transformer_J_No(tnew, sys_no)
        assert j_no >= 0, "transformer j no, please check!"
        return j_no

    # all transformer j no
    def get_transformer_all_j_no(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_j_no(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    # transformer i name
    def get_transformer_i_name(self, tnew, sys_no=-1):
        return self.get_bus_name(self.get_transformer_i_no(tnew, sys_no), sys_no)

    # all transformer i name
    def get_transformer_all_i_name(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        return self.get_bus_all_name(None, sys_no)[self.get_transformer_all_i_no(transformer_list, sys_no)]

    # transformer j name
    def get_transformer_j_name(self, tnew, sys_no=-1):
        return self.get_bus_name(self.get_transformer_j_no(tnew, sys_no), sys_no)

    # all transformer j name
    def get_transformer_all_j_name(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        return self.get_bus_all_name(None, sys_no)[self.get_transformer_all_j_no(transformer_list, sys_no)]

    # transformer asynchronous system no
    def get_transformer_sys_no(self, line_no):
        sys_no = self.__psDLL.get_Transformer_Sys_No(line_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    # all transformer asynchronous system no
    def get_transformer_all_sys_no(self, transformer_list=None):
        transformer_list = np.arange(self.get_transformer_number(), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_sys_no(transformer_no)
        return self.__intBuffer[:len(transformer_list)].astype(np.int32)

    # transformer NO
    def get_transformer_NO(self, transformer_no, sys_no=-1):
        a_NO = self.__psDLL.get_Transformer_No(transformer_no, sys_no)
        assert a_NO >= 0, "transformer No is wrong, please check!"
        return a_NO

    # all transformer NO
    def get_transformer_all_NO(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__intBuffer[index] = self.get_transformer_NO(transformer_no, sys_no)
        return self.__intBuffer[:len(transformer_list)]

    # transformer current capacity
    def get_transformer_current_capacity(self, transformer_no, sys_no=-1):
        current_capacity = self.__psDLL.get_Transformer_Current_Capacity(
            transformer_no, sys_no)
        assert current_capacity > 0.0, "transformer current capacity wrong, please check!"
        return current_capacity

    # all transformer current capacity
    def get_transformer_all_current_capacity(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__doubleBuffer[index] = self.get_transformer_current_capacity(transformer_no, sys_no)
        return self.__doubleBuffer[:len(transformer_list)].astype(np.float32)

    # load flow transformer result, pi, qi, pj, qj
    def get_transformer_lf_result(self, transformer_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Transformer_LF_Result(buffer, transformer_no, sys_no) == True, "get lf transformer result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    # load flow all transformer result, pi, qi, pj, qj
    def get_transformer_all_lf_result(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.get_transformer_lf_result(transformer_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(transformer_list)*4].reshape(len(transformer_list), 4).astype(np.float32)

    # transient stability, current step, transformer, result, pi, qi, pj, qj
    def get_transformer_ts_cur_step_result(self, transformer_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer 
        assert self.__psDLL.get_Transformer_TS_CurStep_Result(buffer, transformer_no, sys_no) == True, "get ts transformer result failed, please check!"
        if rt is True:
            return buffer[:4].astype(np.float32)

    # transient stability, current step, all transformer, result, pi, qi, pj, qj
    def get_transformer_all_ts_cur_step_result(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.get_transformer_ts_cur_step_result(transformer_no, sys_no, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(transformer_list)*4].reshape(len(transformer_list), 4).astype(np.float32)

    # transient stability, step, transformer, result, pi, qi, pj, qj
    def get_transformer_ts_step_result(self, step, transformer_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_transformer_ts_cur_step_result(transformer_no, sys_no)

    # transient stability, current step, all transformer, result, pi, qi, pj, qj
    def get_transformer_all_ts_step_result(self, step, transformer_list=None, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_transformer_all_ts_cur_step_result(transformer_list, sys_no)

    # transient stability, all steps, all transformer, result, pi, qi, pj, qj
    def get_transformer_all_ts_result(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Transformer_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all transformer results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_transformer_number(sys_no)].reshape(total_step, self.get_transformer_number(sys_no), 4)
        return all_result[:, transformer_list, :].astype(np.float32)

    # transformer info
    def get_transformer_info(self, tnew, sys_no=-1):
        return [self.get_transformer_i_name(tnew, sys_no), self.get_transformer_j_name(tnew, sys_no), self.get_transformer_NO(tnew, sys_no)]
    
    # transformer all info
    def get_transformer_all_info(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        return [self.get_transformer_all_i_name(transformer_list, sys_no), self.get_transformer_all_j_name(transformer_list, sys_no), self.get_transformer_all_NO(transformer_list, sys_no)]

    ################################################################################################
    # generators
    ################################################################################################
    # total number of generators
    def get_generator_number(self, sys_no=-1):
        if sys_no == -1:
            return self.__nGenerator
        n_generator = self.__psDLL.get_N_Generator(sys_no)
        assert n_generator >= 0, 'total number of generator wrong, please check!'
        return n_generator

    # generator bus no
    def get_generator_bus_no(self, gnew, sys_no=-1):
        bus_no = self.__psDLL.get_Generator_Bus_No(gnew, sys_no)
        assert bus_no >= 0, "generator i no wrong, please check!"
        return bus_no

    # all generator bus no
    def get_generator_all_bus_no(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__intBuffer[index] = self.get_generator_bus_no(generator_no, sys_no)
        return self.__intBuffer[:len(generator_list)].astype(np.int32)

    # generator bus name
    def get_generator_bus_name(self, gnew, sys_no=-1):
        return self.get_bus_name(self.get_generator_bus_no(gnew, sys_no), sys_no)

    # all generator bus name
    def get_generator_all_bus_name(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        return self.get_bus_all_name(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    # generator asynchronous system no
    def get_generator_sys_no(self, generator_no):
        sys_no = self.__psDLL.get_Generator_Sys_No(generator_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    # all generator asynchronous system no
    def get_generator_all_sys_no(self, generator_list=None):
        generator_list = np.arange(self.get_generator_number(), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__intBuffer[index] = self.get_generator_sys_no(generator_no)
        return self.__intBuffer[:len(generator_list)].astype(np.int32)

    # generator lf bus type
    def get_generator_lf_bus_type(self, gnew, sys_no=-1):
        b_type = self.__psDLL.get_Generator_LF_Bus_Type(gnew, sys_no)
        if b_type == 16:
            return 'slack'
        if b_type == 1:
            return 'pq'
        if b_type == 8:
            return 'pv'
        if b_type == 4:
            return 'pv_pq'
        if b_type == 2:
            return 'pq_pv'
        raise Exception("unknown lf bus type, please check!")

    # all generator lf bus type
    def get_generator_all_lf_bus_type(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        b_type = list()
        for generator_no in generator_list:
            b_type.append(self.get_generator_lf_bus_type(generator_no, sys_no))
        return np.array(b_type)
    
    # all ctrl generator
    def get_generator_all_ctrl(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        all_ctrl = np.arange(self.get_generator_number(None, sys_no)[self.get_generator_all_lf_bus_type(None, sys_no) != 'slack']) if sys_no != -1 else self.__indexCtrlGen
        return generator_list[np.where([gen in all_ctrl for gen in generator_list])]
    
    # all slack generator
    def get_generator_all_slack(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        all_ctrl = np.arange(self.get_generator_number(None, sys_no)[self.get_generator_all_lf_bus_type(None, sys_no) != 'slack']) if sys_no != -1 else self.__indexCtrlGen
        return generator_list[np.where([gen in all_ctrl for gen in generator_list])]

    # generator v set
    def get_generator_v_set(self, gnew, sys_no=-1):
        v_set = self.__psDLL.get_Generator_V0(gnew, sys_no)
        assert v_set > -1.0e10, "generator v_set wrong, please check!"
        return v_set

    # all generator v set
    def get_generator_all_v_set(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_v_set(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # set generator v set
    def set_generator_v_set(self, vset, gnew, sys_no=-1):
        assert self.__psDLL.set_Generator_V0(vset, gnew, sys_no), "set generator v set wrong, please check!"

    # set all generator v set
    def set_generator_all_v_set(self, vset_array, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        assert len(vset_array) == len(generator_list), "generator number mismatch, please check!"
        for (vset, generator_no) in zip(vset_array, generator_list):
            self.set_generator_v_set(vset, generator_no, sys_no)

    # generator p set
    def get_generator_p_set(self, gnew, sys_no=-1):
        p_set = self.__psDLL.get_Generator_P0(gnew, sys_no)
        assert p_set > -1.0e10, "generator p_set wrong, please check!"
        return p_set

    # all generator p set
    def get_generator_all_p_set(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_p_set(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # set generator p set
    def set_generator_p_set(self, pset, gnew, sys_no=-1):
        assert self.__psDLL.set_Generator_P0(pset, gnew, sys_no), "set generator p set wrong, please check!"

    # set all generator p set
    def set_generator_all_p_set(self, pset_array, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        assert len(pset_array) == len(generator_list), "generator number mismatch, please check!"
        for (pset, generator_no) in zip(pset_array, generator_list):
            self.set_generator_p_set(pset, generator_no, sys_no)

    # generator vmax
    def get_generator_vmax(self, generator_no, sys_no=-1):
        vmax = self.get_bus_vmax(self.get_generator_bus_no(generator_no, sys_no), sys_no)
        assert vmax > -1.0e10, "generator vmax wrong, please check!"
        return vmax

    # all generator vmax
    def get_generator_all_vmax(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        return self.get_bus_all_vmax(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    # generator vmin
    def get_generator_vmin(self, generator_no, sys_no=-1):
        vmin = self.get_bus_vmin(self.get_generator_bus_no(generator_no, sys_no), sys_no)
        assert vmin > -1.0e10, "generator vmin wrong, please check!"
        return vmin

    # all generator vmin
    def get_generator_all_vmin(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        return self.get_bus_all_vmin(None, sys_no)[self.get_generator_all_bus_no(generator_list, sys_no)]

    # generator pmax
    def get_generator_pmax(self, generator_no, sys_no=-1):
        pmax = self.__psDLL.get_Generator_PMax(generator_no, sys_no)
        assert pmax > -1.0e10, "generator pmax wrong, please check!"
        # pmax = self.get_generator_p_set(generator_no, sys_no) if abs(pmax) < 1.0e-6 else pmax
        return pmax

    # all generator pmax
    def get_generator_all_pmax(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_pmax(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # generator pmin
    def get_generator_pmin(self, generator_no, sys_no=-1):
        pmin = self.__psDLL.get_Generator_PMin(generator_no, sys_no)
        assert pmin > -1.0e10, "generator pmin wrong, please check!"
        # pmin = self.get_generator_p_set(generator_no, sys_no) if abs(pmin) < 1.0e-6 else pmin
        return pmin

    # all generator pmin
    def get_generator_all_pmin(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_pmin(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # generator qmax
    def get_generator_qmax(self, generator_no, sys_no=-1):
        qmax = self.__psDLL.get_Generator_QMax(generator_no, sys_no)
        assert qmax > -1.0e10, "generator qmax wrong, please check!"
        return qmax

    # all generator qmax
    def get_generator_all_qmax(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_qmax(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # generator qmin
    def get_generator_qmin(self, generator_no, sys_no=-1):
        qmin = self.__psDLL.get_Generator_QMin(generator_no, sys_no)
        assert qmin > -1.0e10, "generator qmin wrong, please check!"
        return qmin

    # all generator qmin
    def get_generator_all_qmin(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_qmin(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # generator tj
    def get_generator_tj(self, generator_no, sys_no=-1):
        tj = self.__psDLL.get_Generator_Tj(generator_no, sys_no)
        assert tj > -1.0e10, "generator tj wrong, please check!"
        return tj

    # all generator tj
    def get_generator_all_tj(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__doubleBuffer[index] = self.get_generator_tj(generator_no, sys_no)
        return self.__doubleBuffer[:len(generator_list)].astype(np.float32)

    # load flow generator result, pg, qg
    def get_generator_lf_result(self, generator_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer            
        assert self.__psDLL.get_Generator_LF_Result(buffer, generator_no, sys_no) == True, "get lf generator result failed, please check!"
        if rt is True:
            return buffer[:2].astype(np.float32)

    # load flow all generator result, pi, qi, pj, qj
    def get_generator_all_lf_result(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.get_generator_lf_result(generator_no, sys_no, self.__doubleBuffer[index*2:], False)
        return self.__doubleBuffer[:len(generator_list)*2].reshape(len(generator_list), 2).astype(np.float32)

    # transient stability, current step, result dimension
    def get_generator_ts_result_dimension(self, generator_no, sys_no=-1, need_inner_e=False):
        dim = self.__psDLL.get_Generator_TS_Result_Dimension(generator_no, sys_no, need_inner_e)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    # transient stability, current step, generator, result, δ, ω, v, θ, pg, qg
    def get_generator_ts_cur_step_result(self, generator_no, sys_no=-1, need_inner_e=False, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_TS_CurStep_Result(buffer, generator_no, sys_no, need_inner_e) == True, "get ts generator result failed, please check!"
        if rt is True:
            buffer[:self.get_generator_ts_result_dimension(generator_no, sys_no, need_inner_e)]

    # transient stability, current step, all generator, result, δ, ω, v, θ, pg, qg
    def get_generator_all_ts_cur_step_result(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.get_generator_ts_cur_step_result(generator_no, sys_no, False, self.__doubleBuffer[index*6:], False)
        return self.__doubleBuffer[:len(generator_list)*6].reshape(len(generator_list), 6).astype(np.float32)

    # transient stability, step, generator, result, δ, ω, v, θ, pg, qg
    def get_generator_ts_step_result(self, step, generator_no, sys_no=-1, need_inner_e=False):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_generator_ts_cur_step_result(generator_no, sys_no, need_inner_e)

    # transient stability, generator, all step
    def get_generator_ts_all_step_result(self, generator_no, sys_no=-1, need_inner_e=False):
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_ts_result_dimension(generator_no, sys_no, need_inner_e)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no)
            self.get_generator_ts_cur_step_result(generator_no, sys_no, need_inner_e, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    # transient stability, current step, all generator, result, δ, ω, v, θ, pg, qg
    def get_generator_all_ts_step_result(self, step, generator_list=None, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_generator_all_ts_cur_step_result(generator_list, sys_no)

    # transient stability, all steps, all generator, result, δ, ω, v, θ, pg, qg
    def get_generator_all_ts_result(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Generator_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all generator results failed!"
        all_result = self.__doubleBuffer[0:total_step * 6 * self.get_generator_number(sys_no)].reshape(total_step, self.get_generator_number(sys_no), 6)
        return all_result[:, generator_list, :].astype(np.float32)
    
    ################################################################################################
    # exciter
    ################################################################################################
    # transient stability, current step, exiter result dimension
    def get_generator_exciter_ts_result_dimension(self, generator_no, sys_no=-1):
        dim = self.__psDLL.get_Generator_Exciter_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    # transient stability, current step, generator, exiter result
    def get_generator_exciter_ts_cur_step_result(self, generator_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_Exciter_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator exiter result failed, please check!"
        if rt is True:
            buffer[:self.get_generator_exciter_ts_result_dimension(generator_no, sys_no)]

    # transient stability, step, generator, exiter result
    def get_generator_exciter_ts_step_result(self, step, generator_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_generator_exciter_ts_cur_step_result(generator_no, sys_no)

    # transient stability, generator, exiter all step
    def get_generator_exciter_ts_all_step_result(self, generator_no, sys_no=-1):
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_exciter_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no)
            self.get_generator_exciter_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # governer
    ################################################################################################
    # transient stability, current step, governer result dimension
    def get_generator_governor_ts_result_dimension(self, generator_no, sys_no=-1):
        dim = self.__psDLL.get_Generator_Governor_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim
    
    # transient stability, current step, generator, governer result
    def get_generator_governor_ts_cur_step_result(self, generator_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Generator_Governor_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator governer result failed, please check!"
        if rt is True:
            buffer[:self.get_generator_governor_ts_result_dimension(generator_no, sys_no)]

    # transient stability, step, generator, governer result
    def get_generator_governor_ts_step_result(self, step, generator_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_generator_governor_ts_cur_step_result(generator_no, sys_no)

    # transient stability, generator, governer all step
    def get_generator_governor_ts_all_step_result(self, generator_no, sys_no=-1):
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_governor_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no)
            self.get_generator_governor_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # pss
    ################################################################################################
    # transient stability, current step, pss result dimension
    def get_generator_pss_ts_result_dimension(self, generator_no, sys_no=-1):
        dim = self.__psDLL.get_Generator_PSS_TS_Result_Dimension(generator_no, sys_no)
        assert dim >= 0, "generator i no wrong, please check!"
        return dim

    # transient stability, current step, pss, pss result
    def get_generator_pss_ts_cur_step_result(self, generator_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_PSS_Governer_TS_CurStep_Result(buffer, generator_no, sys_no) == True, "get ts generator governer result failed, please check!"
        if rt is True:
            buffer[:self.get_generator_pss_ts_result_dimension(generator_no, sys_no)]

    # transient stability, step, generator, pss result
    def get_generator_pss_ts_step_result(self, step, generator_no, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_generator_pss_ts_cur_step_result(generator_no, sys_no)

    # transient stability, generator, pss all step
    def get_generator_pss_ts_all_step_result(self, generator_no, sys_no=-1):
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_generator_pss_ts_result_dimension(generator_no, sys_no)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no)
            self.get_generator_pss_ts_cur_step_result(generator_no, sys_no, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    ################################################################################################
    # load
    ################################################################################################
    # total number of loads
    def get_load_number(self, sys_no=-1):
        if sys_no == -1:
            return self.__nLoad
        n_load = self.__psDLL.get_N_Load(sys_no)
        assert n_load >= 0, 'total number of load wrong, please check!'
        return n_load

    # load bus no
    def get_load_bus_no(self, dnew, sys_no=-1):
        bus_no = self.__psDLL.get_Load_Bus_No(dnew, sys_no)
        assert bus_no >= 0, "load i no wrong, please check!"
        return bus_no

    # all load bus no
    def get_load_all_bus_no(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__intBuffer[index] = self.get_load_bus_no(load_no, sys_no)
        return self.__intBuffer[:len(load_list)].astype(np.int32)

    # load bus name
    def get_load_bus_name(self, dnew, sys_no=-1):
        return self.get_bus_name(self.get_load_bus_no(dnew, sys_no), sys_no)

    # all load bus name
    def get_load_all_bus_name(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        return self.get_bus_all_name(None, sys_no)[self.get_load_all_bus_no(load_list, sys_no)]

    # load asynchronous system no
    def get_load_sys_no(self, load_no):
        sys_no = self.__psDLL.get_Load_Sys_No(load_no)
        assert sys_no >= 0, "asynchronous system detection failed!"
        return sys_no

    # all load asynchronous system no
    def get_load_all_sys_no(self, load_list=None):
        load_list = np.arange(self.get_load_number(), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__intBuffer[index] = self.get_load_sys_no(load_no)
        return self.__intBuffer[:len(load_list)].astype(np.int32)

    # load p set
    def get_load_p_set(self, dnew, sys_no=-1):
        v_set = self.__psDLL.get_Load_P0(dnew, sys_no)
        assert v_set > -1.0e10, "load p_set wrong, please check!"
        return v_set

    # all load p set
    def get_load_all_p_set(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__doubleBuffer[index] = self.get_load_p_set(load_no, sys_no)
        return self.__doubleBuffer[:len(load_list)].astype(np.float32)

    # set load p set
    def set_load_p_set(self, pset, dnew, sys_no=-1):
        assert self.__psDLL.set_Load_P0(pset, dnew, sys_no), "set load p set wrong, please check!"

    # set all load p set
    def set_load_all_p_set(self, pset_array, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        assert len(pset_array) == len(load_list), "load number mismatch, please check!"
        for (pset, load_no) in zip(pset_array, load_list):
            self.set_load_p_set(pset, load_no, sys_no)

    # load q set
    def get_load_q_set(self, dnew, sys_no=-1):
        q_set = self.__psDLL.get_Load_Q0(dnew, sys_no)
        assert q_set > -1.0e10, "load q_set wrong, please check!"
        return q_set

    # all load q set
    def get_load_all_q_set(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__doubleBuffer[index] = self.get_load_q_set(load_no, sys_no)
        return self.__doubleBuffer[:len(load_list)].astype(np.float32)

    # set load q set
    def set_load_q_set(self, qset, dnew, sys_no=-1):
        assert self.__psDLL.set_Load_Q0(qset, dnew, sys_no), "set load p set wrong, please check!"

    # set all load q set
    def set_load_all_q_set(self, qset_array, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        assert len(qset_array) == len(load_list), "load number mismatch, please check!"
        for (qset, load_no) in zip(qset_array, load_list):
            self.set_load_q_set(qset, load_no, sys_no)

    # load flow load result, pl, ql
    def get_load_lf_result(self, load_no, sys_no=-1, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Load_LF_Result(buffer, load_no, sys_no) == True, "get lf load result failed, please check!"
        if rt is True:
            return buffer[:2].astype(np.float32)

    # load flow all load result, pl, ql
    def get_load_all_lf_result(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.get_load_lf_result(load_no, sys_no, self.__doubleBuffer[index*2:], False)
        return self.__doubleBuffer[:len(load_list)*2].reshape(len(load_list), 2).astype(np.float32)

    # transient stability, current step, result dimension
    def get_load_ts_result_dimension(self, load_no, sys_no=-1, need_dynamic_variable=False):
        dim = self.__psDLL.get_Load_TS_Result_Dimension(load_no, sys_no, need_dynamic_variable)
        assert dim >= 0, 'load i no wrong, please check!'
        return dim

    # transient stability, current step, load, result, v, θ, pl, ql
    def get_load_ts_cur_step_result(self, load_no, sys_no=-1, need_dynamic_variable=False, buffer=None, rt=True):
        buffer = self.__doubleBuffer if buffer is None else buffer
        assert self.__psDLL.get_Load_TS_CurStep_Result(buffer, load_no, sys_no, need_dynamic_variable) == True, "get ts load result failed, please check!"
        if rt is True:
            return buffer[:self.get_load_ts_result_dimension(load_no, sys_no, need_dynamic_variable)].astype(np.float32)

    # transient stability, current step, all load, result, v, θ, pl, ql
    def get_load_all_ts_cur_step_result(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.get_load_ts_cur_step_result(load_no, sys_no, False, self.__doubleBuffer[index*4:], False)
        return self.__doubleBuffer[:len(load_list)*4].reshape(len(load_list), 4).astype(np.float32)

    # transient stability, step, load, result, v, θ, pl, ql
    def get_load_ts_step_result(self, step, load_no, sys_no=-1, need_dynamic_variable=False):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_load_ts_cur_step_result(load_no, sys_no, need_dynamic_variable)

    # transient stability, all step, load, result
    def get_load_ts_all_step_result(self, load_no, sys_no=-1, need_dynamic_variable=False):
        steps = self.get_info_ts_finish_step() + 1
        dim = self.get_load_ts_result_dimension(load_no, sys_no, need_dynamic_variable)
        for step in range(steps):
            self.set_info_ts_step_element_state(step, sys_no)
            self.get_load_ts_cur_step_result(load_no, sys_no, need_dynamic_variable, self.__doubleBuffer[step*dim:], False)
        return self.__doubleBuffer[:steps*dim].reshape(steps, dim).astype(np.float32)

    # transient stability, current step, all load, result, v, θ, pl, ql
    def get_load_all_ts_step_result(self, step, load_list=None, sys_no=-1):
        self.set_info_ts_step_element_state(step, sys_no)
        return self.get_load_all_ts_cur_step_result(load_list, sys_no)

    # transient stability, all steps, all load, result, v, θ, pl, ql
    def get_load_all_ts_result(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        self.get_info_ts_finish_step()
        assert self.__cur_total_step >= 0, 'transient simulation not done yet!'
        total_step = self.__cur_total_step + 1
        assert self.__psDLL.get_Load_TS_All_Result(self.__doubleBuffer, sys_no), "get ts all load results failed!"
        all_result = self.__doubleBuffer[0:total_step * 4 * self.get_load_number(sys_no)].reshape(total_step, self.get_load_number(sys_no), 4)
        return all_result[:, load_list, :].astype(np.float32)

    ################################################################################################
    # network
    ################################################################################################
    # number of non zero element in admittance matrix
    def get_n_non_zero(self, sys_no=-1):
        if sys_no == -1:
            return self.__nNonzero
        n_non = self.__psDLL.get_N_Non_Zero_Element(sys_no)
        assert n_non >= 0, 'total number of non-zero element wrong, please check!'
        return n_non

    # number of non zero element in inverse admittance matrix
    def get_n_inverse_non_zero(self, sys_no=-1):
        if sys_no == -1:
            return self.__nInverseNonZero
        n_non = self.__psDLL.get_N_Inverse_Non_Zero_Element(sys_no)
        assert n_non >= 0, 'total number of inverse non-zero element wrong, please check!'
        return n_non

    # check connectivity, return the number of asynchronous system
    def get_n_acsystem_check_connectivity(self, ts_step=0, sys_no=-1):
        self.set_info_ts_step_element_state(ts_step)
        n_acsystem = self.__psDLL.get_N_ACSystem_Check_Connectivity(sys_no)
        assert n_acsystem >= 0, "ac system no. is not correct, please check!"
        return n_acsystem

    # ac line, connectivity
    def get_acline_connectivity(self, lnew, sys_no=-1):
        return self.__psDLL.get_ACLine_Connectivity(lnew, sys_no)

    # ac line, all connectivity
    def get_acline_all_connectivity(self, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        for (index, acline_no) in zip(range(len(acline_list)),acline_list):
            self.__boolBuffer[index] = self.get_acline_connectivity(acline_no, sys_no)
        return self.__boolBuffer[:len(acline_list)].astype(np.bool)

    # set, ac line, connectivity
    def set_acline_connectivity(self, cmark, lnew, sys_no=-1):
        assert self.__psDLL.set_ACLine_Connectivity(cmark, lnew, sys_no), "set acline connectivity mark wrong, please check!"

    # set, ac line, all connectivity
    def set_acline_all_connectivity(self, cmarks, acline_list=None, sys_no=-1):
        acline_list = np.arange(self.get_acline_number(sys_no), dtype=np.int) if acline_list is None else acline_list
        assert len(acline_list) == len(cmarks), "marks length does not match, please cleck"
        for (cmark, acline_no) in zip(cmarks, acline_list):
            self.set_acline_connectivity(cmark, acline_no, sys_no)

    # transformer, connectivity
    def get_transformer_connectivity(self, lnew, sys_no=-1):
        return self.__psDLL.get_Transformer_Connectivity(lnew, sys_no)

    # transformer, all connectivity
    def get_transformer_all_connectivity(self, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        for (index, transformer_no) in zip(range(len(transformer_list)), transformer_list):
            self.__boolBuffer[index] = self.get_transformer_connectivity(transformer_no, sys_no)
        return self.__boolBuffer[:len(transformer_list)].astype(np.bool)

    # set, transformer, connectivity
    def set_transformer_connectivity(self, cmark, lnew, sys_no=-1):
        assert self.__psDLL.set_Transformer_Connectivity(cmark, lnew, sys_no), "set transformer connectivity mark wrong, please check!"

    # set, transformer, all connectivity
    def set_transformer_all_connectivity(self, cmarks, transformer_list=None, sys_no=-1):
        transformer_list = np.arange(self.get_transformer_number(sys_no), dtype=np.int) if transformer_list is None else transformer_list
        assert len(transformer_list) == len(cmarks), "marks length does not match, please cleck"
        for (cmark, transformer_no) in zip(cmarks, transformer_list):
            self.set_transformer_connectivity(cmark, transformer_no, sys_no)

    # rebuild all network data
    def set_rebuild_all_network_data(self):
        assert self.__psDLL.set_Rebuild_All_Network_Data() is True, "rebuild network data failed, please check"

    # generator, connectivity
    def get_generator_connectivity(self, gnew, sys_no=-1):
        return self.__psDLL.get_Generator_Connectivity(gnew, sys_no)

    # generator, all connectivity
    def get_generator_all_connectivity(self, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        for (index, generator_no) in zip(range(len(generator_list)), generator_list):
            self.__boolBuffer[index] = self.get_generator_connectivity(generator_no, sys_no)
        return self.__boolBuffer[:len(generator_list)].astype(np.bool)

    # set, generator, connectivity
    def set_generator_connectivity(self, cmark, gnew, sys_no=-1):
        assert self.__psDLL.set_Generator_Connectivity(cmark, gnew, sys_no), "set generator connectivity mark wrong, please check!"

    # set, generator, all connectivity
    def set_generator_all_connectivity(self, cmarks, generator_list=None, sys_no=-1):
        generator_list = np.arange(self.get_generator_number(sys_no), dtype=np.int) if generator_list is None else generator_list
        assert len(generator_list) == len(cmarks), "marks length does not match, please cleck!"
        for (cmark, generator_no) in zip(cmarks, generator_list):
            self.set_generator_connectivity(cmark, generator_no, sys_no)
    
    # load, connectivity
    def get_load_connectivity(self, dnew, sys_no=-1):
        return self.__psDLL.get_Load_Connectivity(dnew, sys_no)

    # load, all connectivity
    def get_load_all_connectivity(self, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        for (index, load_no) in zip(range(len(load_list)), load_list):
            self.__boolBuffer[index] = self.get_load_connectivity(load_no, sys_no)
        return self.__boolBuffer[:len(load_list)].astype(np.bool)

    # set, load, connectivity
    def set_load_connectivity(self, cmark, dnew, sys_no=-1):
        assert self.__psDLL.set_Load_Connectivity(cmark, dnew, sys_no), "set load connectivity mark wrong, please check!"

    # set, load, all connectivity
    def set_load_all_connectivity(self, cmarks, load_list=None, sys_no=-1):
        load_list = np.arange(self.get_load_number(sys_no), dtype=np.int) if load_list is None else load_list
        assert len(load_list) == len(cmarks), "marks length does not match, please cleck!"
        for (cmark, load_no) in zip(cmarks, load_list):
            self.set_load_connectivity(cmark, load_no, sys_no)

    # full admittance matrix
    def get_admittance_matrix_full(self, ts_step=0, sys_no=-1):
        self.set_info_ts_step_network_state(ts_step)
        assert self.__psDLL.get_Admittance_Matrix_Full(
            self.__doubleBuffer, sys_no), "get full admittance matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        return self.__doubleBuffer[:2*n_bus*n_bus].reshape(2, n_bus, n_bus).astype(np.float32)

    # impedance matrix, naturally full
    def get_impedance_matrix_full(self, ts_step=0, sys_no=-1):
        self.set_info_ts_step_network_state(ts_step)
        assert self.__psDLL.get_Impedence_Matrix_Full(
            self.__doubleBuffer, sys_no), "get impedance matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        return self.__doubleBuffer[:2*n_bus*n_bus].reshape(2, n_bus, n_bus).astype(np.float32)

    # factorized impedance matrix
    def get_impedance_matrix_factorized(self, ts_step=0, sys_no=-1):
        self.set_info_ts_step_network_state(ts_step)
        assert self.__psDLL.get_Impedence_Matrix_Factorized(
            self.__doubleBuffer, sys_no), "get factorized inverse matrix failed, please check!"
        n_bus = self.get_bus_number(sys_no)
        n_invnonzero = self.get_n_inverse_non_zero(sys_no)
        f_inv = list()
        f_inv.append(self.__doubleBuffer[0:(
            n_invnonzero+n_bus)*6].reshape(n_invnonzero+n_bus, 6).astype(np.float32))
        f_inv.append(self.__doubleBuffer[(n_invnonzero+n_bus)*6:2*(
            n_invnonzero+n_bus)*6].reshape(n_invnonzero+n_bus, 6).astype(np.float32))
        f_inv.append(self.__doubleBuffer[2*(n_invnonzero+n_bus)*6:2*(
            n_invnonzero+n_bus)*6+n_bus*6].reshape(n_bus, 6).astype(np.float32))
        return np.array(f_inv, dtype=object)

    ################################################################################################
    # fault and disturbance
    ################################################################################################
    # clear all fault and disturbance
    def set_fault_disturbance_clear_all(self):
        assert self.__psDLL.set_Fault_Disturbance_Clear_All() is True, "clear fault and disturbance failed, please check!"

    # add acline fault, 0-three phase fault, 1-three phase disconnection
    def set_fault_add_acline(self, fault_type, fault_dis, start_time, end_time, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Fault(
            fault_type, fault_dis, start_time, end_time, 0, ele_pos, False, sys_no), "add fault acline failed, please check!"

    # change adline fault, 0-three phase fault, 1-three phase disconnection
    def set_fault_change_acline(self, fault_type, fault_dis, start_time, end_time, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Fault(
            fault_type, fault_dis, start_time, end_time, 0, ele_pos, True, sys_no), "change fault acline failed, please check!"

    # add transformer fault, 0-three phase fault, 1-three phase disconnection
    def set_fault_add_transformer(self, fault_type, fault_dis, start_time, end_time, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Fault(
            fault_type, fault_dis, start_time, end_time, 1, ele_pos, False, sys_no), "add fault transformer failed, please check!"

    # change transformer fault, 0-three phase fault, 1-three phase disconnection
    def set_fault_change_transformer(self, fault_type, fault_dis, start_time, end_time, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Fault(
            fault_type, fault_dis, start_time, end_time, 1, ele_pos, True, sys_no), "change fault transformer failed, please check!"

    # add generator fault, 0-tripping
    def set_disturbance_add_generator(self, dis_type, dis_time, dis_per, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Disturbance(
            dis_type, dis_time, dis_per, 0, ele_pos, False, sys_no), "add disturbance generator failed, please check!"

    # change generator fault, 0-tripping
    def set_disturbance_change_generator(self, dis_type, dis_time, dis_per, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Disturbance(
            dis_type, dis_time, dis_per, 0, ele_pos, True, sys_no), "change disturbance generator failed, please check!"

    # add load fault, 0-tripping
    def set_disturbance_add_load(self, dis_type, dis_time, dis_per, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Disturbance(
            dis_type, dis_time, dis_per, 1, ele_pos, False, sys_no), "add disturbance load failed, please check!"

    # change load fault, 0-tripping
    def set_disturbance_change_load(self, dis_type, dis_time, dis_per, ele_pos, sys_no=-1):
        assert self.__psDLL.set_Fault_Disturbance_Add_Disturbance(
            dis_type, dis_time, dis_per, 1, ele_pos, True, sys_no), "change disturbance load failed, please check!"

    ################################################################################################
    # integrated function
    ################################################################################################
    # resume topology
    def set_topology_original(self, sys_no=-1):
        acline_cmarks = np.full(self.get_acline_number(sys_no), True)
        self.set_acline_all_connectivity(acline_cmarks, None, sys_no)
        transformer_cmarks = np.full(self.get_transformer_number(sys_no), True)
        self.set_transformer_all_connectivity(transformer_cmarks, None, sys_no)
        generator_cmarks = np.full(self.get_generator_number(sys_no), True)
        self.set_generator_all_connectivity(generator_cmarks, None, sys_no)
        load_cmarks = np.full(self.get_load_number(sys_no), True)
        self.set_load_all_connectivity(load_cmarks, None, sys_no)

    # topology sampler, change topology, keep connectivity
    def get_topology_sample(self, topo_change=1, sys_no=-1):
        if topo_change == 0:
            self.set_topology_original()
            return None
        acline_no = np.arange(self.get_acline_number(sys_no))
        # acline_no = np.array([1,2,3,4,7,8,9,10,11,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
        n_sample = 0
        selected_no = None
        while True:
            selected_no = self.__rng.choice(acline_no, size=topo_change, replace=False)
            for line_no in selected_no:
                self.set_acline_connectivity(True, line_no, sys_no)
            if self.get_n_acsystem_check_connectivity() == self.__nACSystem:
                break
            for line_no in selected_no:
                self.set_acline_connectivity(False, line_no, sys_no)
            n_sample += 1
            assert n_sample < 100, "topology sample failed, please check!"
        self.set_rebuild_all_network_data()
        print([[line_no, self.get_acline_info(line_no, sys_no)] for line_no in selected_no])
        return [[line_no, self.get_acline_info(line_no, sys_no)] for line_no in selected_no]

    # pf resume original status
    def set_pf_original_status(self):
        self.set_generator_all_v_set(self.__generator_v_origin)
        self.set_generator_all_p_set(self.__generator_p_origin)
        self.set_load_all_p_set(self.__load_p_origin)
        self.set_load_all_q_set(self.__load_q_origin)
    
    # get pf original status
    def get_pf_original_status(self):
        return np.concatenate((self.__generator_v_origin, self.__generator_p_origin[self.__indexCtrlGen], self.__load_p_origin, self.__load_q_origin))

    # pf status setting, sample = array[gen_v, ctrl_gen_p, load_p, load_q]
    def set_pf_initiation(self, 
                      sample, 
                      generator_v_list=None,
                      generator_p_list=None,
                      load_p_list=None,
                      load_q_list=None,
                      sys_no=-1):
        sample_part = sample.copy()
        # gen v set
        if generator_v_list is None:
            generator_v_list = np.arange(self.get_generator_number(sys_no))
        self.set_generator_all_v_set(sample_part[:len(generator_v_list)], generator_v_list, sys_no)
        sample_part = sample_part[len(generator_v_list):]
        # gen p set
        if generator_p_list is None:
            generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        self.set_generator_all_p_set(sample_part[:len(generator_p_list)], generator_p_list, sys_no)
        sample_part = sample_part[len(generator_p_list):]
        # load p set
        if load_p_list is None:
            load_p_list = np.arange(self.get_load_number(sys_no))
        self.set_load_all_p_set(sample_part[:len(load_p_list)], load_p_list, sys_no)
        sample_part = sample_part[len(load_p_list):]
        # load q set
        if load_q_list is None:
            load_q_list = np.arange(self.get_load_number(sys_no))
        self.set_load_all_q_set(sample_part[:len(load_q_list)], load_q_list, sys_no)

    # get bounds
    def get_pf_bounds(self, 
                      generator_v_list=None,
                      generator_p_list=None,
                      load_p_list=None,
                      load_q_list=None,
                      load_max=None,
                      load_min=None,
                      sys_no=-1):
        # gen v max&min
        if generator_v_list is None: generator_v_list = np.arange(self.get_generator_number(sys_no))
        gen_vmax = self.get_generator_all_vmax(generator_v_list, sys_no)
        gen_vmin = self.get_generator_all_vmin(generator_v_list, sys_no)
        # gen p max&min
        if generator_p_list is None: generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        gen_pmax = self.get_generator_all_pmax(generator_p_list, sys_no)
        gen_pmin = self.get_generator_all_pmin(generator_p_list, sys_no)
        if np.all(gen_pmax == 0) and np.all(gen_pmin == 0):
            if sys_no != -1:
                for i in range(sys_no):
                    generator_p_list += self.get_bus_number(i)
            gen_pmax = self.__generator_p_origin[generator_p_list] * self.__generator_p_bounds[1]
            gen_pmin = self.__generator_p_origin[generator_p_list] * self.__generator_p_bounds[0]
        # load bound
        load_max = self.__load_pq_bounds[1] if load_max is None else load_max
        load_min = self.__load_pq_bounds[0] if load_min is None else load_min
        # load p max&min
        if load_p_list is None: load_p_list = np.arange(self.get_load_number(sys_no))
        if load_max == -1 and load_min == -1:# do not change current load settings, pmax = pmin = current p set
            load_pmax = load_pmin = self.get_load_all_p_set(load_p_list, sys_no)
        else:
            if sys_no != -1:
                for i in range(sys_no):
                    load_p_list += self.get_bus_number(i)
            load_p_set = self.__load_p_origin[load_p_list]
            load_pmax = load_p_set * load_max
            load_pmin = load_p_set * load_min
        # load q max&min
        if load_q_list is None: load_q_list = np.arange(self.get_load_number(sys_no))
        if load_max == -1 and load_min == -1:# do not change current load settings, qmax = qmin = current q set
            load_qmax = load_qmin = self.get_load_all_q_set(load_q_list, sys_no)
        else:
            if sys_no != -1:
                for i in range(sys_no):
                    load_p_list += self.get_bus_number(i)
            load_q_set = self.__load_q_origin[load_q_list]
            load_qmax = load_q_set * load_max
            load_qmin = load_q_set * load_min
        # concatenate bounds
        lower = np.concatenate((gen_vmin, gen_pmin, load_pmin, load_qmin))
        upper = np.concatenate((gen_vmax, gen_pmax, load_pmax, load_qmax))
        idx = lower > upper
        lower[idx], upper[idx] = upper[idx], lower[idx]
        assert np.any(lower > upper) == False, "get lf bounds failed, please check!"
        return [lower, upper]

    # pf cal and status check
    def get_pf_status_check(self):
        if self.cal_pf_basic_power_flow_nr():
            slack_p = self.get_generator_all_lf_result(self.__indexSlack)[:, 0]
            if np.all(slack_p > 0.0):
                lf_v = self.get_bus_all_lf_result()[:, 0]
                if np.any(lf_v > self.get_bus_all_vmax()) or np.any(lf_v < self.get_bus_all_vmin()):
                    return [True, True, False]
                else:
                    return [True, True, True]
            else:
                return [True, False, False]
        else:
            return [False, False, False]

    # pf sampler, gen_v, ctrl_gen_p, load_p, load_q
    def get_pf_sample_simple_random(self, 
                                    num=1,
                                    generator_v_list=None,
                                    generator_p_list=None,
                                    load_p_list=None,
                                    load_q_list=None,
                                    load_max=None,
                                    load_min=None,
                                    sys_no=-1,
                                    check_converge=True,
                                    check_slack=True,
                                    check_voltage=True
                                    ):
        [lower_bounds, upper_bounds] = self.get_pf_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        bound_size = len(lower_bounds)
        sample_buffer = list()
        for _ in range(num):
            [converge, valid_slack, valid_v] = [False, False, False]
            counting = 0
            while (False in [converge, valid_slack, valid_v]):
                r_vector = self.__rng.random(bound_size)
                cur_status = lower_bounds + (upper_bounds - lower_bounds) * r_vector
                self.set_pf_initiation(cur_status, generator_v_list, generator_p_list, load_p_list, load_q_list)
                # total_load = sum(self.get_load_all_p_set())
                # total_slack = sum(self.get_generator_all_pmax(self.__indexSlack))
                # total_ctrl = sum(self.get_generator_all_p_set(self.__indexCtrlGen))
                # if total_load < total_ctrl or total_load > total_ctrl + total_slack:
                #     continue
                [converge, valid_slack, valid_v] = self.get_pf_status_check()
                [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                if False not in [converge, valid_slack, valid_v]:
                    break
            sample_buffer.append(cur_status)       
        return sample_buffer
        
    def get_pf_sample_stepwise(self, 
                               num=1,
                               generator_v_list=None,
                               generator_p_list=None,
                               load_p_list=None,
                               load_q_list=None,
                               load_max=None,
                               load_min=None,
                               sys_no=-1,
                               check_converge=True,
                               check_slack=True,
                               check_voltage=True
                               ):
        if generator_v_list is None: generator_v_list = np.arange(self.get_generator_number(sys_no))
        if generator_p_list is None: generator_p_list = self.get_generator_all_ctrl(None, sys_no)
        if load_p_list is None: load_p_list = np.arange(self.get_load_number(sys_no))
        if load_q_list is None: load_q_list = np.arange(self.get_load_number(sys_no))
        [lower_bounds, upper_bounds] = self.get_pf_bounds(generator_v_list, generator_p_list, load_p_list, load_q_list, load_max, load_min, sys_no)
        gen_pmax = upper_bounds[len(generator_v_list):len(generator_v_list)+len(generator_p_list)]
        gen_pmin = lower_bounds[len(generator_v_list):len(generator_v_list)+len(generator_p_list)]
        slack_max = self.get_generator_all_pmax(self.__indexSlack)
        slack_min = self.get_generator_all_pmin(self.__indexSlack)
        load_pmax = upper_bounds[len(generator_v_list)+len(generator_p_list):len(generator_v_list)+len(generator_p_list)+len(load_p_list)]
        load_pmin = lower_bounds[len(generator_v_list)+len(generator_p_list):len(generator_v_list)+len(generator_p_list)+len(load_p_list)]
        load_qmax = upper_bounds[-len(load_q_list):]
        load_qmin = lower_bounds[-len(load_q_list):]
        sample_buffer = list()
        gen_v = self.get_generator_all_v_set(generator_v_list, sys_no)
        gen_p = np.zeros(len(generator_p_list))
        for _ in range(num):
            # load p max&min
            load_psum = -1.0
            while load_psum < (sum(gen_pmin) + sum(slack_min)) or load_psum > (sum(gen_pmax) + sum(slack_max)):
                load_p = load_pmin + (load_pmax - load_pmin) * self.__rng.random(len(load_pmax))
                load_psum = sum(load_p)
            # load q max&min
            load_q = load_qmin + (load_qmax - load_qmin) * self.__rng.random(len(load_qmax))
            # gen p
            [converge, valid_slack, valid_v] = [False, False, False]
            while (False in [converge, valid_slack, valid_v]):
                gen_p.fill(0.)
                gen_order = self.__rng.choice(np.arange(len(gen_p)), len(generator_p_list), replace=False)
                load_psum = sum(load_p)
                for i in range(len(gen_p)):
                    gen_no = gen_order[i]
                    remain_gen = gen_order[i+1:] if i+1 < len(gen_p) else []
                    pmin = max(load_psum - sum(gen_pmax[remain_gen]) - sum(slack_max), gen_pmin[gen_no])
                    pmax = min(load_psum - sum(gen_pmin[remain_gen]) - sum(slack_min), gen_pmax[gen_no])                    
                    gen_p[gen_no] = pmin + (pmax - pmin) * self.__rng.random()
                    load_psum -= gen_p[gen_no]
                cur_status = np.concatenate([gen_v, gen_p, load_p, load_q])
                self.set_pf_initiation(cur_status)
                [converge, valid_slack, valid_v] = self.get_pf_status_check()
                [converge, valid_slack, valid_v] = [x or y for x, y in zip([converge, valid_slack, valid_v], [not check_converge, not check_slack, not check_voltage])]
                if False not in [converge, valid_slack, valid_v]:
                    break
            sample_buffer.append(cur_status)
        return sample_buffer
    
    # ts, check stability
    def check_stability(self, maximum_delta=180.):
        std_result = self.get_acsystem_all_ts_result()[0]
        if std_result[:, 1].max() < maximum_delta:
            fin_step = self.get_info_ts_finish_step()
            if fin_step + 1 != self.get_info_ts_max_step():
                print(fin_step, std_result[:, 1].max())
                # raise Exception("stable and early finish, please check!")
            return True
        else:
            return False

    # ts checker
    def get_ts_sample_check(self, pf_samples):
        
        for sample in pf_samples:
            self.set_pf_initiation(sample)

        # self.get_generator_

        # self.ts_initiation()
        # dim = 2
        # load_choice = [0.7, 0.8, 0.9, 1.0, 1.1]
        # if dpt == -1:
        #     load_level = self.__rng.choice(load_choice)
        #     decrypt = self.__rng.choice(self.__nStateGrid) // 5
        # else:
        #     assert dpt < self.__nStateGrid, 'dpt larger than grid_no'
        #     assert dpt >= 0, 'dpt smaller than 0'
        #     load_level = load_choice[dpt % len(load_choice)]
        #     decrypt = dpt // len(load_choice)
        # start_gen_v = self.__nGen
        # lower_gen_v = self.get_gen_v_lower().copy()
        # upper_gen_v = self.get_gen_v_upper().copy()
        # delta_v = (upper_gen_v - lower_gen_v) / dim
        # start_ctrl_p = self.get_n_ctrl_gen()
        # lower_ctrl_p = self.get_ctrl_gen_p_min().copy()
        # upper_ctrl_p = self.get_ctrl_gen_p_max().copy()
        # delta_ctrl_p = (upper_ctrl_p - lower_ctrl_p) / dim
        # n_direction = start_gen_v + start_ctrl_p
        # assert n_direction == self.__nDirection, 'direction num do not match'
        # norm_load_p = self.__origin_Load_P
        # delta_p = norm_load_p * 0.1
        # norm_load_q = self.__origin_Load_Q
        # delta_q = norm_load_q * 0.1
        # encrypt = np.ones((n_direction,), dtype=int)
        # for i in range(n_direction):
        #     encrypt[i] = decrypt % dim
        #     decrypt = decrypt / dim
        # gen_v = lower_gen_v + \
        #     ((upper_gen_v - lower_gen_v) / dim) * encrypt[0:start_gen_v]
        # gen_v += delta_v * self.__rng.random(len(delta_v))
        # ctrl_p = lower_ctrl_p + \
        #     ((upper_ctrl_p - lower_ctrl_p) / dim) * \
        #     encrypt[start_gen_v:n_direction]
        # ctrl_p += delta_ctrl_p * self.__rng.random(len(delta_ctrl_p))
        # load_p = norm_load_p * load_level
        # load_p += delta_p * self.__rng.random(len(norm_load_p))
        # load_q = norm_load_q * load_level
        # load_q += delta_q * self.__rng.random(len(norm_load_p))
        # self.set_gen_v0(gen_v)
        # self.set_ctrl_gen_p(ctrl_p)
        # self.set_load_p(load_p)
        # self.set_load_q(load_q)
        # # converge, valid slack, valid v
        # if self.cal_pf():
        #     slack_p = self.get_pf_slack_p()
        #     if slack_p >= 0:
        #         lf_v = self.get_pf_v()
        #         too_large = lf_v > self.get_v_upper()
        #         too_small = lf_v < self.get_v_lower()
        #         if (True in too_large) or (True in too_small):
        #             return True, True, False
        #         else:
        #             return True, True, True
        #     else:
        #         return True, False, False
        # else:
        #     return False, False, False

    def output_pf(self):
        output_file_name = self.__absFilePath + '/results/'
        if not os.path.exists(output_file_name):
            os.makedirs(output_file_name)
        output_file_name += self.__absFileName + '_' + \
            str(self.__flg) + '_' + self.__timeStamp + '.lot'
        assert self.__psDLL.pf_OutputPowerFlow(
            output_file_name, len(output_file_name)), 'output pf failure!'

    def set_fault_disturbance(self, fault_no):
        assert fault_no < self.__nBus + 2 * self.__nACLine, 'fault no out of range!'
        self.__psDLL.scan_SetFaultDisturbance(fault_no, -1)

    def catch_ts_y_fault(self):
        assert self.__cur_total_step != -1, 'transient simulation not done yet!'
        self.__doubleBuffer.fill(0.)
        self.__psDLL.scan_GetFaultAdmittance(self.__doubleBuffer)
        y_fault = self.__doubleBuffer[0:self.__nBus * self.__nBus * 2 * 2].copy()\
            .reshape(4, self.__nBus, self.__nBus).astype(np.float32)
        return y_fault

    def get_curves(self, curves_set, cut_length=None):
        curve = list()
        std_result = self.catch_ts_std_result()
        gen_result = self.catch_ts_gen_delta()
        bus_result = self.catch_ts_bus_detail()
        branch_result = self.catch_ts_branch_detail()
        for curve_name in curves_set:
            # y origin
            # z origin
            # y fault
            if curve_name == 'y_fault':
                curve.append(self.catch_ts_y_fault())
            # auto analysis
            elif curve_name == 'auto_ana':
                curve.append(np.transpose(std_result))
            elif curve_name == 'time':
                curve.append(std_result[:, 0])
            elif curve_name == 'max_delta':
                curve.append(std_result[:, 1])
            elif curve_name == 'min_freq':
                curve.append(std_result[:, 2])
            elif curve_name == 'max_freq':
                curve.append(std_result[:, 3])
            elif curve_name == 'min_vol':
                curve.append(std_result[:, 4])
            elif curve_name == 'max_vol':
                curve.append(std_result[:, 5])
            # gen delta
            elif curve_name == 'all_delta':
                curve.append(np.transpose(gen_result))
            elif 'delta_' in curve_name:
                gen_name = curve_name.split('_')[1]
                for gen_no in range(self.get_n_gen()):
                    if gen_name == self.get_gen_name(gen_no):
                        curve.append(gen_result[:, gen_no])
                        break
            # bus detail
            elif curve_name == 'all_bus_detail':
                curve.append(bus_result)
            elif 'bus_detail_' in curve_name:
                bus_name = curve_name.split('_')[1]
                for bus_no in range(self.get_bus_number()):
                    if bus_name == self.get_bus_name(bus_no):
                        curve.append(bus_result[:, bus_no, :])
                        break
            elif 'vol_' in curve_name:
                bus_name = curve_name.split('_')[1]
                for bus_no in range(self.get_bus_number()):
                    if bus_name == self.get_bus_name(bus_no):
                        curve.append(bus_result[:, bus_no, 0])
                        break
            # branch detail
            elif curve_name == 'all_branch_detail':
                curve.append(np.transpose(branch_result, (0, 2, 1)))
            elif 'ac_line_detail_' in curve_name:
                ac_line_info = curve_name.split('_')[3:]
                for line_no in range(self.get_n_ac_line()):
                    if ac_line_info == self.get_ac_line_info(line_no):
                        curve.append(branch_result[:, line_no, 0])
                        break
            elif 'p_ac_line_' in curve_name:
                ac_line_info = curve_name.split('_')[3:]
                ac_line_info[-1] = int(ac_line_info[-1])
                for line_no in range(self.get_n_ac_line()):
                    if ac_line_info == self.get_ac_line_info(line_no):
                        curve.append(branch_result[:, line_no, 0])
                        break
            elif 'p_transformer_' in curve_name:
                transformer_info = curve_name.split('_')[2:]
                transformer_info[-1] = int(transformer_info[-1])
                for tran_no in range(self.get_n_transformer()):
                    if transformer_info == self.get_transformer_info(tran_no):
                        curve.append(
                            branch_result[:, tran_no + self.get_n_ac_line(), 0])
                        break
        if cut_length is not None:
            for idx in range(len(curve)):
                curve[idx] = curve[idx][:cut_length]
        return curve

    def output_ts(self):
        output_file_name = self.__absFilePath + '/results/'
        if not os.path.exists(output_file_name):
            os.makedirs(output_file_name)
        output_file_name += self.__absFileName + '_' + \
            str(self.__flg) + '_' + self.__timeStamp + '.nsot'
        assert self.__psDLL.ts_Scanning_NodalShortCircuit(
            output_file_name, len(output_file_name)), 'output ts failure!'

    def remove_line(self, line_no):
        self.__psDLL.remove_ACLine(line_no)

    def resume_line(self, line_no):
        self.__psDLL.resume_ACLine(line_no)

    def reset_pf(self):
        self.set_gen_v0(self.__origin_Gen_V)
        self.set_ctrl_gen_p(self.__origin_Gen_P[self.__indexCtrlGen])
        self.set_load_p(self.__origin_Load_P)
        self.set_load_q(self.__origin_Load_Q)

    def set_flg(self, flg):
        self.__flg = flg

    def set_random_state(self, rng):
        self.__rng = rng

    def set_seed(self, seed):
        self.__rng = np.random.default_rng(seed)

    def get_n_state_grid(self):
        return self.__nStateGrid

    def get_n_direction(self):
        return self.__nDirection

    def get_all_ac_line_info(self):
        ac_lines_info = list()
        for i in range(self.__nACLine):
            ac_lines_info.append(self.get_ac_line_info(i))
        return ac_lines_info

    def get_ac_line_info(self, line_new):
        tmp = self.__psDLL.get_ACLine_IName(line_new, -1)
        i_name = string_at(tmp, -1).decode('gbk')
        tmp = self.__psDLL.get_ACLine_JName(line_new, -1)
        j_name = string_at(tmp, -1).decode('gbk')
        num = self.__psDLL.get_ACLine_No(line_new, -1)
        return [i_name, j_name, num]

    def get_all_transformer_info(self):
        transformers_info = list()
        for i in range(self.__nTransformer):
            transformers_info.append(self.get_transformer_info(i))
        return transformers_info

    def get_transformer_info(self, transformer_new):
        tmp = self.__psDLL.get_Transformer_IName(transformer_new, -1)
        i_name = string_at(tmp, -1).decode('gbk')
        tmp = self.__psDLL.get_Transformer_JName(transformer_new, -1)
        j_name = string_at(tmp, -1).decode('gbk')
        num = self.__psDLL.get_Transformer_No(transformer_new, -1)
        return [i_name, j_name, num]

    def set_ctrl_gen_p(self, gen_p):
        input_p = self.__origin_Gen_P.copy()
        input_p[self.__indexCtrlGen] = gen_p
        assert self.__nGen == self.__psDLL.set_Gen_P(
            input_p, self.__nGen, - 1), 'Set Gen P error!'

    def get_pf_slack_p(self):
        assert self.__nGen == self.__psDLL.get_LF_GenP(self.__doubleBuffer, self.__bufferSize, -1), \
            'Read LF result Slack P error!'
        return self.__doubleBuffer[self.__indexSlack]

    def get_pf_ctrl_gen_p(self):
        assert self.__nGen == self.__psDLL.get_LF_GenP(self.__doubleBuffer, self.__bufferSize, -1), \
            'Read LF result Gen P'
        return self.__doubleBuffer[self.__indexCtrlGen]

    def _buffer_tests(self):
        # basic tests
        self.__psDLL.get_LF_V(self.__doubleBuffer, self.__bufferSize, -1)
        x = list(self.__doubleBuffer)
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)
        self.__doubleBuffer[0] = 9999.
        x[0] = 1234
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)
        self.__psDLL.get_LF_V(self.__doubleBuffer, self.__bufferSize, -1)
        print(self.__doubleBuffer)
        print(list(self.__doubleBuffer))
        print(x)


if __name__ == '__main__':
    # """
    start_time = datetime.datetime.now()
    api = psopsAPI()
    api.check_stability()
    print(api.get_pf_sample_simple_random(load_max=-1, load_min=-1))

