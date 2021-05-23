"""
Modified from https://github.com/modestyachts/ARS/blob/master/code/ars.py
"""
from ast import dump
from numba.core.types.npytypes import NumpyFlatType
from numpy.core.arrayprint import dtype_is_implied
import ray
import collections
import pickle

from ray.worker import init
from gym_psops.envs import opf_Env
from gym_psops.envs import sopf_Env
from gym_psops.envs.psops import psopsAPI
import numpy as np
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


font1 = {'size': 12}

def plot_dots(x, y, color, lw=1):
    assert x.shape[0] == y.shape[0], "x, y dimensional error"
    if y.ndim == 1:
        plt.scatter(x, y, alpha=0.5)
    elif y.ndim == 2:
        pass
    else:
        raise Exception(f"{y.ndim} dimensional data input!")

def plot_show():
    plt.show()


def contour_plot(xx, yy, zz):
    contour = plt.contour(xx, yy, zz, 100, cmap='rainbow')
    plt.clabel(contour, fontsize=6, colors='k')
    # 去掉坐标轴刻度
    # plt.xticks(())
    # plt.yticks(())
    # 填充颜色，f即filled,6表示将三色分成三层，cmap那儿是放置颜色格式，hot表示热温图（红黄渐变）
    # 更多颜色图参考：https://blog.csdn.net/mr_cat123/article/details/80709099
    # 颜色集，6层颜色，默认的情况不用写颜色层数,
    # c_set = plt.contourf(xx, yy, zz, cmap='rainbow')
    # or c_map='hot'
    # 设置颜色条，（显示在图片右边）
    # plt.colorbar(c_set)
    # 显示
    plt.show()


def plot_3d(xx, yy, zz):
    fig = plt.figure()
    ax = Axes3D(fig)
    # ax.plot_surface(xx, yy, zz, cmap='rainbow')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1,
                    alpha=0.75, cmap='rainbow')
    ax.contour(xx, yy, zz, zdir='z', offset=zz.min(),
               cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
    ax.contour(xx, yy, zz, zdir='y', offset=yy.max(), cmap='rainbow')
    ax.contour(xx, yy, zz, zdir='x', offset=xx.min(), cmap='rainbow')
    plt.show()


def read_result(f_path):
    file = open(f_path, 'r', encoding='gbk')
    lines = file.readlines()
    file.close()
    num = 0
    gen1 = list()
    gen2 = list()
    list_loss = list()
    list_180 = list()
    list_500 = list()
    list_d = list()
    for line in lines:
        if line[0] == 'g':
            continue
        if line.strip() != '':
            num += 1
            line_array = line.strip().split()
            gen1.append(float(line_array[0]))
            gen2.append(float(line_array[1]))
            list_loss.append(-float(line_array[2]))
            list_180.append(float(line_array[3]))
            list_500.append(float(line_array[4]))
            list_d.append(float(line_array[5]))
    mesh = int(num ** 0.5)
    assert mesh * mesh == num, 'Line num is not correct!'
    gen1 = np.array(gen1).reshape(mesh, mesh)
    gen2 = np.array(gen2).reshape(mesh, mesh)
    list_loss = np.array(list_loss).reshape(mesh, mesh)
    list_180 = np.array(list_180).reshape(mesh, mesh)
    list_500 = np.array(list_500).reshape(mesh, mesh)
    list_d = np.array(list_d).reshape(mesh, mesh)
    # list_d = (360. - list_d) / (360. + list_d) * 100.
    # list_d[list_d > 300.] = 300.
    # list_loss = list_180
    list_loss = -list_loss + list_180 - 10.
    loss_min = list_loss[list_d < 180.].max()
    idx = np.where(list_loss == loss_min)
    print(list_loss[list_d < 180.].max(), gen1[idx], gen2[idx])
    list_loss = np.array(list_loss)
    # list_loss = list_d
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_surface(gen2, gen1, list_loss, rstride=1, cstride=1, alpha=0.5, cmap='rainbow', zorder=0)
    # # ax.contour(gen2, gen1, list_loss, zdir='z', offset=list_loss.min(), cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
    # # ax.set_xlabel('Gen_BUS38', fontdict=font1)
    # # ax.set_ylabel('Gen_BUS30', fontdict=font1)
    # # ax.set_zlabel('Delta_MAX', fontdict=font1)
    # ax.plot_surface(gen1, gen2, list_loss, rstride=1, cstride=1, alpha=0.5, cmap='rainbow', zorder=0)
    # ax.contour(gen1, gen2, list_loss, zdir='z', offset=list_loss.min(), cmap='rainbow')  # 等高线图，要设置offset，为Z的最小值
    # ax.set_xlabel('Gen_BUS30', fontdict=font1)
    # ax.set_ylabel('Gen_BUS38', fontdict=font1)
    # ax.set_zlabel('Ts', fontdict=font1)
    # plt.tick_params(labelsize=12)
    # ls = np.load('./result_dist.npz', allow_pickle=True)
    # x = ls['action_sequence']
    # print(x.shape)
    # for search_line in x:
    #     xd = search_line[:, 0]
    #     yd = search_line[:, 1]
    #     zd = search_line[:, 2]
    #     ax.scatter3D(xd, yd, zd, c='k', marker='o', s=50, linewidths=1, alpha=0.25, zorder=1)
    #     ax.plot(xd, yd, zd, c='k', lw=2, alpha=0.2, zorder=1)
    # ax.scatter3D(xd, yd, zd, c='k', marker='o', s=50, linewidths=5, zorder=2)  # 绘制散点图
    # ax.plot(xd, yd, zd, c='k', lw=2, zorder=2)
    contour = plt.contour(gen1, gen2, list_loss, 100, cmap='rainbow', zorder=2)
    plt.clabel(contour, fontsize=8, colors='k')
    plt.xlim((1.25, 3.75))
    plt.ylim((3.25, 9.75))
    plt.xlabel('Gen_BUS30', fontdict=font1)
    plt.ylabel('Gen_BUS38', fontdict=font1)
    plt.tick_params(labelsize=12)
    ls = np.load('../results/result_dist.npz', allow_pickle=True)
    x = ls['action_sequence']
    print(x.shape)
    max_re = -999999999.9
    max_no = -1
    max_seq = None
    for seq_no in range(x.shape[0]):
        if max_re < np.max(x[seq_no][:, 3]):
            max_re = np.max(x[seq_no][:, 3])
            max_no = seq_no
            max_seq = x[seq_no]
    assert max_seq is not None, 'no max reward'
    print(max_re, max_no, max_seq)
    xd = max_seq[:7, 1]
    yd = max_seq[:7, 2]
    plt.scatter(xd, yd, c='k', marker='o', s=50, linewidths=5, zorder=2)
    plt.plot(xd, yd, c='k', lw=2, zorder=2)
    for search_line in x:
        xd = search_line[:, 1]
        yd = search_line[:, 2]
        plt.scatter(xd, yd, c='darkgrey', marker='o', s=50, linewidths=1, alpha=0.25, zorder=1)
        plt.plot(xd, yd, c='darkgrey', lw=2, alpha=0.25, zorder=1)
    # plt.scatter(xd, yd, c='r', marker='o', s=50, linewidths=5, zorder=2)
    # plt.plot(xd, yd, c='r', lw=2, zorder=2)
    # search_line = x[-2]
    # xd = search_line[:, 1]   
    # yd = search_line[:, 2]
    # plt.scatter(xd, yd, c='b', marker='o', s=50, linewidths=5, zorder=2)
    # plt.plot(xd, yd, c='b', lw=2, zorder=2)
    plt.show()
    # plot_3d(gen1, gen2, np.array(list_loss))
    # contour_plot(gen1, gen2, np.array(list_loss))
    # plot_3d(gen1, gen2, np.array(list_180))
    # contour_plot(gen1, gen2, np.array(list_180))
    # plot_3d(gen1, gen2, np.array(list_500))
    # contour_plot(gen1, gen2, np.array(list_500))
    # plot_3d(gen2, gen1, list_d)
    # contour_plot(gen2, gen1, list_d)
    # plot_3d(gen2, gen1, 1. / list_d)
    # contour_plot(gen2, gen1, 1. / list_d)


class SampleGenerator:
    def __init__(self, worker_no, total_worker):
        self.workerNo = worker_no
        self.totalWorker = total_worker
        self.__rng = np.random.default_rng(4242 + self.workerNo * 64)
        # self.__rng = np.random.default_rng()
        self.__api = psopsAPI(flg=self.workerNo, rng=self.__rng)
        # self.env = sopf_Env(rng=self.__rng, test_on=True)
        # self.env.set_flg(worker_no)

    def set_seed(self, seed):
        self.__rng = np.random.default_rng(4242 + seed * 64)
        self.__api.set_random_state(self.__rng)
    
    # i,j did not distinguished
    def simple_random_sampling(self, 
                               num_disconnect, 
                               num_pf, 
                               sample_round, 
                               cut=100, 
                               check_slack=True, 
                               check_voltage=True):
        api = self.__api
        cut += 1
        # resume topo
        api.set_topology_original()
        # sample result
        sample_dict = collections.OrderedDict()
        # bus name
        if sample_round == 0 and self.workerNo == 0:
            sample_dict['bus_name'] = api.get_bus_all_name()
        # topo set
        if (num_disconnect != 0):
            topo_samples = sample_dict['w_{}_r_{}_topo'.format(self.workerNo, sample_round)] = api.get_topology_sample(num_disconnect)
            line_disconnect = [ts[0] for ts in topo_samples]
        else:
            sample_dict['w_{}_r_{}_topo'.format(self.workerNo, sample_round)] = None
            line_disconnect = None
        # y, z, factor z init
        sample_dict['w_{}_r_{}_y_init'.format(self.workerNo, sample_round)] = api.get_admittance_matrix_full(0)
        sample_dict['w_{}_r_{}_z_init'.format(self.workerNo, sample_round)] = api.get_impedance_matrix_full(0)
        sample_dict['w_{}_r_{}_factor_z_init'.format(self.workerNo, sample_round)] = api.get_impedance_matrix_factorized(0)
        # ========================================================================================================================================================================
        # start pf looping
        n_acline = api.get_acline_number()
        stable_sample = 0
        unstable_sample = 0
        pf_samples = api.get_pf_sample_simple_random(num=num_pf, check_slack=check_slack, check_voltage=check_voltage)
        # pf_samples = api.get_pf_sample_stepwise(num=num_pf, check_slack=check_slack, check_voltage=check_voltage)
        for pf, pf_no in zip(pf_samples, range(len(pf_samples))):
            sample_dict['w_{}_r_{}_pf_{}_set'.format(self.workerNo, sample_round, pf_no)] = pf#############################################################################################################
            api.set_pf_initiation(pf)
            # --------------------------------------------------------------------------------------------------------------------------------------------------------------------
            # ts loop
            for acline_no in range(n_acline):
                # avoiding disconnected line
                if line_disconnect is not None and acline_no in line_disconnect:
                    continue
                # check connectivity
                api.set_acline_connectivity(False, acline_no)
                if api.get_acsystem_number() == api.get_n_acsystem_check_connectivity(0):
                    api.set_rebuild_all_network_data()
                    # check load flow after fault clearing
                    if (api.cal_pf_basic_power_flow_nr()):########################################################################################################################################################################
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_fault_info'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_acline_info(acline_no)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_y_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_admittance_matrix_full(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_z_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_full(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_factor_z_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_factorized(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_pf_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_bus_all_lf_result()
                    else:
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_fault_info'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_acline_info(acline_no)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_y_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_admittance_matrix_full(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_z_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_full(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_factor_z_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_factorized(0)
                        sample_dict['w_{}_r_{}_pf_{}_fault_{}_pf_clear'.format(self.workerNo, sample_round, pf_no, acline_no)] = np.zeros([api.get_bus_number(), 6], dtype=np.float32)
                    # ts calculation
                    api.set_acline_connectivity(True, acline_no)
                    api.set_rebuild_all_network_data()
                    api.cal_pf_basic_power_flow_nr()
                    # side i
                    api.set_fault_disturbance_clear_all()########################################################################################################################################################################
                    api.set_fault_add_acline(0, 0, 0.0, 0.1, acline_no)
                    api.set_fault_add_acline(1, 0, 0.1, 10., acline_no)
                    ########################################################################################################################################################################
                    ts_step = api.cal_ts_simulation_ti_sv()########################################################################################################################################################################
                    # api.get_generator_all_ts_result([0])########################################################################################################################################################################
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_stable'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.check_stability()
                    if sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_stable'.format(self.workerNo, sample_round, pf_no, acline_no)] == True:
                        stable_sample += 1
                    else:
                        unstable_sample += 1
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_autoAna'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_acsystem_all_ts_result()[0, :cut, :]
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_allBus'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_bus_all_ts_result()[:cut, :, :]
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_y_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_admittance_matrix_full(1)
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_z_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_full(1)
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_i_ts_factor_z_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_factorized(1)
                    # side j
                    api.set_fault_disturbance_clear_all()
                    api.set_fault_add_acline(0, 100, 0.0, 0.1, acline_no)
                    api.set_fault_add_acline(1, 100, 0.1, 10., acline_no)
                    ts_step = api.cal_ts_simulation_ti_sv()
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_stable'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.check_stability()
                    if sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_stable'.format(self.workerNo, sample_round, pf_no, acline_no)] == True:
                        stable_sample += 1
                    else:
                        unstable_sample += 1
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_autoAna'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_acsystem_all_ts_result()[0, :cut, :]
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_allBus'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_bus_all_ts_result()[:cut, :, :]
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_y_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_admittance_matrix_full(1)
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_z_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_full(1)
                    sample_dict['w_{}_r_{}_pf_{}_fault_{}_j_ts_factor_z_fault'.format(self.workerNo, sample_round, pf_no, acline_no)] = api.get_impedance_matrix_factorized(1)
                else:
                    api.set_acline_connectivity(True, acline_no)
        
        # print('data generation finished.')
        print('number of stable vs unstable: {} vs {}'.format(stable_sample, unstable_sample))
        location = '../results/20210516_cut0_nolimit/'
        if not os.path.exists(location):
            os.makedirs(location)
        f = open(location + 'sample_{}_{}.sp'.format(self.workerNo, sample_round), 'wb')
        pickle.dump(sample_dict, f)
        f.close()
        return [stable_sample, unstable_sample]
    
    def ts_sampler_simple_random(self, num=1, check_slack=True, check_voltage=True):
        api = self.__api
        aclines = np.arange(api.get_acline_number())
        gen_name = api.get_generator_all_bus_name()
        curves = list()
        for i in range(num):
            api.get_pf_sample_simple_random(check_slack=check_slack, check_voltage=check_voltage)
            acline_no = self.__rng.choice(aclines)
            api.set_fault_add_acline(0, 0, 0.0, 0.1, acline_no)
            api.set_fault_add_acline(1, 0, 0.1, 10., acline_no)
            api.cal_ts_simulation_ti_sv()
            curves.append(api.get_generator_all_ts_result()[:, :, 0])
        curves = np.array(curves, dtype=object)
        np.savez('../results/sample_for_mark.npz', gen_name=gen_name, curves=curves)


@ray.remote
class RayWorkerForSampleGenerator(SampleGenerator):
    def __init__(self, worker_no, total_worker):
        super().__init__(worker_no, total_worker)

    def rigid_grid_search(self):
        # 39
        t1 = time.time()
        pos1 = 0
        pos2 = 1
        grid = 10
        gen1 = self.env.get_psops().get_ctrl_gen_p()[pos1]
        gen1 = np.arange(gen1 * 0.5, gen1 * 1.5, gen1 / grid)
        gen2 = self.env.get_psops().get_ctrl_gen_p()[pos2]
        gen2 = np.arange(gen2 * 0.5, gen2 * 1.5, gen2 / grid)
        gen = self.env.get_psops().get_ctrl_gen_p().copy()
        print(gen1, gen2)
        trace_path = "./trace.DAT"
        # trace = open(trace_path, "w")
        # trace.write('gen1\t\tgen2\t\treward\t\tloss\t\tt_180\t\tt_500\t\tmax_d\n')
        list_reward = list()
        list_loss = list()
        list_180 = list()
        list_500 = list()
        list_d = list()
        jump_180 = 0.
        jump_500 = 0.
        jump_d = 99999999.
        for i in range(grid):
            p1 = gen1[i]
            gen[pos1] = p1
            for j in range(grid):
                p2 = gen2[j]
                gen[pos2] = p2
                self.env.set_ctrl(gen[[pos1, pos2]])
                reward = self.env.step(0.)[1]
                list_reward.append(reward)
                loss = (np.sum(self.env.get_psops().get_pf_load_p(), axis=0)
                        - np.sum(self.env.get_psops().get_pf_gen_p(), axis=0))
                list_loss.append(loss)
                # """
                self.env.get_psops().ts_initiation()
                self.env.get_psops().set_fault_disturbance(55)
                total_step = self.env.get_psops().cal_ts()
                """
                result = self.env.get_psops().catch_ts_std_result()
                t_180 = result[total_step - 1][0]
                t_500 = result[total_step - 1][0]
                max_d = result[:, 1].max()
                for k in range(total_step):
                    if t_180 >= result[k, 0] and result[k, 1] >= 180.:
                        t_180 = result[k, 0]
                    if t_500 >= result[k, 0] and result[k, 1] >= 500.:
                        t_500 = result[k, 0]
                        break  # """
                # trace.write(str(p1) + '\t\t' + str(p2) + '\t\t' + str(reward) + '\t\t' + str(loss) + '\t\t'
                #             + str(t_180) + '\t\t' + str(t_500) + '\t\t' + str(max_d)
                #             + '\n')
                # """
                # list_180.append(t_180)
                # list_500.append(t_500)
                # list_d.append(max_d)  # """
            print('round {} finished.'.format(i))
        # trace.close()
        t1 = time.time() - t1
        print('cal time: {} seconds'.format(t1))
        for i in range(grid * grid):
            if list_180[i] > jump_180 and list_180[i] < 10.:
                jump_180 = list_180[i]
            if list_500[i] > jump_500 and list_500[i] < 10.:
                jump_500 = list_500[i]
            if list_d[i] < jump_d and list_d[i] >= 180.:
                jump_d = list_d[i]
        print(jump_180)
        print(jump_500)
        print(jump_d)
        gen2, gen1 = np.meshgrid(gen2, gen1)
        list_reward = np.array(list_reward).reshape(grid, grid)
        list_loss = np.array(list_loss).reshape(grid, grid)
        list_180 = np.array(list_180).reshape(grid, grid)
        list_500 = np.array(list_500).reshape(grid, grid)
        list_d = np.array(list_d).reshape(grid, grid)
        plot_3d(gen1, gen2, np.array(list_reward))
        contour_plot(gen1, gen2, np.array(list_reward))
        plot_3d(gen1, gen2, np.array(list_loss))
        contour_plot(gen1, gen2, np.array(list_loss))
        plot_3d(gen1, gen2, np.array(list_180))
        contour_plot(gen1, gen2, np.array(list_180))
        plot_3d(gen1, gen2, np.array(list_500))
        contour_plot(gen1, gen2, np.array(list_500))
        plot_3d(gen2, gen1, np.array(list_d))
        contour_plot(gen2, gen1, np.array(list_d))
        plot_3d(gen2, gen1, 1. / np.array(list_d))
        contour_plot(gen2, gen1, 1. / np.array(list_d))

    def n_1_ac_line(self):
        n_bus = self.env.get_psops().get_bus_number()
        n_ac_line = self.env.get_psops().get_n_ac_line()
        sim_result = list()
        self.env.get_psops().ts_initiation()
        for fault_no in range(n_bus, n_bus + n_ac_line * 2):
            self.env.get_psops().set_fault_disturbance(fault_no)
            print(self.env.get_psops().get_ac_line_info(
                int((fault_no - n_bus) / 2)))
            total_step = self.env.get_psops().cal_ts()
            result = self.env.get_psops().catch_ts_std_result(total_step)
            step = 0
            for step in range(total_step):
                if result[step][1] >= 180.:
                    # print(fault_no, self.env.get_psops().get_ac_line_info(int((fault_no - n_bus) / 2)),
                    #       ("head" if int((fault_no - n_bus)) % 2 == 0 else "tail"))
                    sim_result.append([result[step][0], ])
                    break
            sim_result.append([fault_no, result[step][0], result[step][1]])
        print("search finished.")
        return sim_result

    def n_1_tables(self, fault_table):
        api = self.env.get_psops()
        n_bus = api.get_bus_number()
        n_ac_line = api.get_n_ac_line()
        n_transformer = api.get_n_transformer()
        if -1 in fault_table:
            fault_table = range(n_bus, n_bus + n_ac_line *
                                2 + n_transformer * 2)
        sim_result = list()
        api.ts_initiation()
        for fault_no in fault_table:
            api.set_fault_disturbance(fault_no)
            total_step = api.cal_ts()
            result = api.catch_ts_std_result(total_step)
            step = 0
            for step in range(total_step):
                if result[step][1] >= 180.:
                    # print(fault_no, self.env.get_psops().get_ac_line_info(int((fault_no - n_bus) / 2)),
                    #       ("head" if int((fault_no - n_bus)) % 2 == 0 else "tail"))
                    break
            sim_result.append([fault_no, api.get_ac_line_info(
                int((fault_no - n_bus) / 2)), result[step][0], result[step][1]])
        return np.array(sim_result)

    def grid_sampler(self, phase_no=0, total_phase=1, end_sample=-1):
        results = []
        n_bus = self.env.get_psops().get_bus_number()
        if self.workerNo == 0 and phase_no == 0:
            results.append(self.env.get_psops().get_bus_all_name())
            results.append(self.env.get_psops().cal_y())
            results.append(self.env.get_psops().cal_z())
        n_state_grid = self.env.get_psops().get_n_state_grid()
        # print (n_state_grid)
        if end_sample == -1:
            n_total_sample = n_state_grid
        elif end_sample * total_phase > n_state_grid:
            n_total_sample = end_sample * total_phase
        else:
            assert end_sample > 0, 'sample num error'
            n_total_sample = end_sample * total_phase
        total_worker = self.totalWorker * total_phase
        quotient = n_total_sample // total_worker
        remainder = n_total_sample % total_worker
        worker_no = self.workerNo * total_phase + phase_no
        start_n = quotient * worker_no + min(worker_no, remainder)
        stop_n = quotient * (worker_no + 1) + min(worker_no + 1, remainder)
        if self.workerNo == 0:
            print('worker {} sample range {} to {}.'.format(
                self.workerNo, start_n, stop_n))
        quotient = n_state_grid // n_total_sample
        remainder = n_state_grid % n_total_sample
        for i in range(start_n, stop_n):
            sample_start = quotient * i + min(i, remainder)
            sample_stop = quotient * (i + 1) + min(i + 1, remainder)
            pf_flags = [False, False, False]
            pf_result = None
            sample_no = -1
            if quotient >= 1:
                for _ in range(5):
                    sample_no = int(np.random.rand() *
                                    (sample_stop - sample_start) + sample_start)
                    pf_flags = self.pf_sampler(sample_no)
                    if False not in pf_flags:
                        pf_result = self.env.get_psops().catch_pf_results()
                        break
            if False in pf_flags:
                while False in pf_flags:
                    pf_flags = self.pf_sampler()
                pf_result = self.env.get_psops().catch_pf_results()
            if self.workerNo == 0:
                print('worker {} sample no {} grid range {} to {}, grid no {}'.format(self.workerNo, i, sample_start,
                                                                                      sample_stop, sample_no))
            results.append(pf_result)
            self.env.get_psops().ts_initiation()
            for fault_no in range(n_bus):
                self.env.get_psops().set_fault_disturbance(fault_no)
                results.append(results[0][fault_no])
                total_step = self.env.get_psops().cal_ts()
                results.append(total_step)
                result = self.env.get_psops().catch_ts_y_fault()
                results.append(result)
                result = self.env.get_psops().catch_ts_std_result(total_step)
                results.append(result[np.arange(33), :])
                results.append(result[total_step - 1, :])
                if result[:, 1].max() < 180.:
                    results.append('stable')
                    if total_step < 1003:
                        print('stable yet not converge', total_step, result)
                else:
                    results.append('unstable')
                result = self.env.get_psops().catch_ts_gen_delta(total_step)
                results.append(result[np.arange(33), :])
                results.append(result[total_step - 1, :])
                result = self.env.get_psops().catch_ts_bus_detail(total_step)
                results.append(result[np.arange(33), :])
                results.append(result[total_step - 1, :])
            if self.workerNo == 0:
                print('Round {} finished.'.format(i))
        results = np.array(results, dtype=object)
        return results

    def grid_sampler_for_mark_gt(self):
        np.random.seed(4242)
        num_sample = 5000  # actual num_sample num is num_sample * 2, half stable, half unstable
        inset = list()
        inset.append('max_delta')
        inset.append('min_freq')
        inset.append('max_freq')
        inset.append('min_vol')
        inset.append('max_vol')
        for gen_no in range(self.env.get_psops().get_n_gen()):
            inset.append('delta_' + self.env.get_psops().get_gen_name(gen_no))
        for bus_no in range(self.env.get_psops().get_bus_number()):
            inset.append('vol_' + self.env.get_psops().get_bus_name(bus_no))
        for line_no in range(self.env.get_psops().get_n_ac_line()):
            line_info = self.env.get_psops().get_ac_line_info(line_no)
            inset.append(
                'p_ac_line_' + line_info[0] + '_' + line_info[1] + '_' + str(line_info[2]))
        for tran_no in range(self.env.get_psops().get_n_transformer()):
            tran_info = self.env.get_psops().get_transformer_info(tran_no)
            inset.append(
                'p_transformer_' + tran_info[0] + '_' + tran_info[1] + '_' + str(tran_info[2]))

        sample_name = list()
        curves = list()
        labels = list()
        curve_length = list()
        stable_num = 0
        unstable_num = 0
        while stable_num < num_sample or unstable_num < num_sample:
            stable_num = 0
            unstable_num = 0
            stable_num, unstable_num = self.get_simulation_result(sample_name=sample_name,
                                                                  curves=curves,
                                                                  curves_set=inset,
                                                                  labels=labels,
                                                                  curve_length=curve_length,
                                                                  stable_num=stable_num,
                                                                  stable_limit=num_sample,
                                                                  unstable_num=unstable_num,
                                                                  unstable_limit=num_sample,)
            print(stable_num, unstable_num)
            if stable_num > 30 and unstable_num > 30:
                break
            else:
                sample_name.clear()
                curves.clear()
                curve_length.clear()
                labels.clear()
                stable_num = 0
                unstable_num = 0

        print('data generation finished. start merging data...')
        curve_length = min(curve_length)
        curves_list = list()
        for case_curve in curves:
            sel_data = np.transpose(np.array(case_curve)[:, 0:curve_length])
            # sel_data = case_curve
            # curves_list.append(sel_data.tolist())
            curves_list.append(sel_data)
        # curves_list = np.array(curves_list, dtype=object).astype(np.float32)
        # print('input curves shape: {}'.format(curves_list.shape))
        print('number of unstable vs stable: {} vs {}'.format(
            labels.count(1), labels.count(0)))
        np.savez('./sample.npz', signal=inset, name=sample_name,
                 curve=curves_list, label=labels)

    def grid_sampler_for_gcn(self, num_pf, sample_round=None):
        api = self.env.get_psops()
        #topological change
        section = 2
        topo_change = None
        if sample_round != 0 and sample_round % section == 0:
            topo_change = sample_round / section
            if api.get_ac_line_info(topo_change) == ['BUS-16', 'BUS-19', 22]:
                return
        for line in topo_change:
            api.remove_line(line)
        api.rebuild_all_network_data()
        # 

        line_info = None
        if sample_round != 0:
            line_info = api.get_ac_line_info(sample_round - 1)
            if line_info == ['BUS-16', 'BUS-19', 22]:
                return
        # bus name/number
        bus_name = self.env.get_psops().get_bus_all_name() 
        for i in range(self.env.get_psops().get_n_ac_line()):
            # ac line change
            # y, inv info
            # get n samples
            for j in range(self.env.get_psops().get_n_ac_line()):
                # i != j
                # set fault
                # 
                pass
            pass
        
        # y init
        y_init = self.env.get_psops().cal_y()

        inset = list()
        inset.append('y_fault')
        inset.append('time')
        inset.append('all_bus_detail')

        sample_name = list()
        curves = list()
        labels = list()
        n_pf = 0
        while n_pf < num_pf:
            stable_num, unstable_num = self.get_simulation_result(sample_name=sample_name,
                                                                  curves=curves,
                                                                  curves_set=inset,
                                                                  labels=labels,
                                                                  cut_length=31)
            print(stable_num, unstable_num)
            n_pf += 1

        print('data generation finished.')
        print('number of unstable vs stable: {} vs {}'.format(
            labels.count(1), labels.count(0)))
        if sample_round is None:
            np.savez('./sample.npz', bus_name=bus_name, y_init=y_init, signal=inset,
                     sample_name=sample_name, curve=curves, label=labels)
        else:
            if sample_round == 0 and self.workerNo == 0:
                np.savez('./sample_{}_{}.npz'.format(sample_round, self.workerNo),
                         bus_name=bus_name, signal=inset, y_init=y_init,
                         sample_name=sample_name, curve=curves, label=labels)
            else:
                np.savez('./sample_{}_{}.npz'.format(sample_round, self.workerNo),
                         sample_name=sample_name, curve=curves, label=labels)
        # change topo back
        for line in topo_change:
            api.resume_line(line)
        api.rebuild_all_network_data()

    def grid_sampler_for_st_gcn(self, num_pf, sample_round=None):
        # bus name/number
        bus_name = self.env.get_psops().get_bus_all_name() 
        # y init
        y_init = self.env.get_psops().cal_y()

        # output list
        inset = list()
        inset.append('y_fault')
        inset.append('time')
        inset.append('all_bus_detail')

        sample_name = list()
        curves = list()
        labels = list()
        n_pf = 0
        while n_pf < num_pf:
            stable_num, unstable_num = self.get_simulation_result(sample_name=sample_name,
                                                                  curves=curves,
                                                                  curves_set=inset,
                                                                  labels=labels,
                                                                  cut_length=31)
            print(stable_num, unstable_num)
            n_pf += 1

        print('data generation finished.')
        print('number of unstable vs stable: {} vs {}'.format(
            labels.count(1), labels.count(0)))
        if sample_round is None:
            np.savez('./sample.npz', bus_name=bus_name, y_init=y_init, signal=inset,
                     sample_name=sample_name, curve=curves, label=labels)
        else:
            if sample_round == 0 and self.workerNo == 0:
                np.savez('./sample_{}_{}.npz'.format(sample_round, self.workerNo),
                         bus_name=bus_name, signal=inset, y_init=y_init,
                         sample_name=sample_name, curve=curves, label=labels)
            else:
                np.savez('./sample_{}_{}.npz'.format(sample_round, self.workerNo),
                         sample_name=sample_name, curve=curves, label=labels)

    def all_critical_line_check(self, line_table, fault_table):
        api = self.env.get_psops()
        for line_no in range(api.get_n_ac_line()):
            cur_line = api.get_ac_line_info(line_no)
            if cur_line in line_table:
                self.critical_line_check(line_no, fault_table)
        for tran_no in range(api.get_n_transformer()):
            cur_tran = api.get_transformer_info(tran_no)
            if cur_tran in line_table:
                self.critical_line_check(
                    api.get_n_ac_line() + tran_no, fault_table)

    def critical_line_check(self, line_no, fault_table):
        env = self.env
        api = env.get_psops()
        env.resume_initial()
        api.cal_pf()
        api.ts_initiation()
        line_info = api.get_ac_line_info(line_no)
        p_origin = abs(api.get_pf_ac_line_p()[line_no])
        r_test = self.n_1_tables(fault_table)
        print(line_info, 0, p_origin)
        print(r_test)
        p_current = p_origin
        search_round = 0
        while False not in r_test[:, 3] < 180.:
            search_round += 1
            pf_flags = [False, False, False]
            p_test = -p_current
            while False in pf_flags or p_test < p_current or p_test > p_current * 1.01:
                pf_flags = self.pf_sampler()
                p_test = abs(api.get_pf_ac_line_p()[line_no])
            print(line_info, search_round, p_test)
            p_current = p_test
            r_test = self.n_1_tables(fault_table)
            print(r_test)

    def get_simulation_result(self,
                              sample_name,  # sample name, power flow name + fault name
                              labels,  # stable or not
                              curves_set,  # what needs to be stored in curves list
                              curves,  # curves list
                              fault_table=None,
                              curve_length=None,  # the data length of curves
                              stable_num=None,  # num of stable samples
                              stable_limit=None,  # num limit of stable samples
                              unstable_num=None,  # num of unstable samples
                              unstable_limit=None,  # num limit of unstable samples
                              cut_length=None
                              ):
        # state check
        assert (stable_num is None and stable_limit is None) or \
               (stable_num is not None and stable_limit is not None), \
            'stable count error!'
        assert (unstable_num is None and unstable_limit is None) or \
               (unstable_num is not None and unstable_limit is not None), \
            'unstable count error!'
        # get api
        api = self.env.get_psops()

        # generate power flow
        pf_flags = [False, False, False]
        while False in pf_flags:
            pf_flags = api.pf_sampler()
        v = api.get_pf_gen_v().copy()
        pg = api.get_pf_ctrl_gen_p().copy()
        pl = api.get_pf_load_p().copy()
        ql = api.get_pf_load_q().copy()
        obs = np.concatenate((v, pg, pl, ql))
        pf_name = 'pf'
        for ob in obs:
            pf_name += '_' + str(ob)

        # start n_1 scanning
        api.ts_initiation()
        n_bus = api.get_bus_number()
        n_ac_line = api.get_n_ac_line()
        no_stable = 0
        no_unstable = 0
        if fault_table is None:
            f_table = range(n_ac_line * 2)
        else:
            f_table = fault_table
        for l_no in f_table:
            line_no = int(l_no / 2)
            # get current line info
            line_info = api.get_ac_line_info(line_no)
            # for ieee-39, skip bus16_bus19
            if line_info == ['BUS-16', 'BUS-19', 22]:
                continue
            # get case name and store it in sample name list
            case_name = pf_name + '_' + line_info[0] + '_' + line_info[1] + '_'
            if l_no % 2 == 0:
                case_name += 'head'
            else:
                case_name += 'tail'
            sample_name.append(case_name)
            # cal fault no
            fault_no = n_bus + l_no
            # fault_no = -1
            api.set_fault_disturbance(fault_no)
            # cal ts
            total_step = api.cal_ts()
            if curve_length is not None:
                curve_length.append(total_step)
            # check stability
            stability = api.check_stability()
            if stability:
                assert total_step == api.get_max_step(), 'Stable yet non-convergent smample exist'
                if stable_num is not None:
                    if stable_num >= stable_limit:
                        continue
                    else:
                        stable_num += 1
                labels.append(0)
                no_stable += 1
            else:
                if unstable_num is not None:
                    if unstable_num >= unstable_limit:
                        continue
                    else:
                        unstable_num += 1
                labels.append(1)
                no_unstable += 1
            # get results
            curves.append(api.get_curves(
                curves_set=curves_set, cut_length=cut_length))
        if stable_num is None:
            return no_stable, no_unstable
        else:
            return stable_num, unstable_num


if __name__ == '__main__':
    sampler = RayWorkerForSampleGenerator(0, 1)
    sampler.grid_sampler_for_mark_gt()
