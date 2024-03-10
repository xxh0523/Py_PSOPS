from unittest import TestProgram
from sample_generator import SampleGenerator, plot_dots, plot_show
from py_psops import Py_PSOPS
from sample_generator import RayWorkerForSampleGenerator, sample_generator
from sample_generator import read_result
import numpy as np
from multiprocessing.pool import Pool
import ray
import time
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


def grid_sampler_for_st_gcn():
    test_worker = RayWorkerForSampleGenerator(0, 1)
    test_worker.grid_sampler_for_st_gcn()

    
if __name__ == '__main__':
    ct = -time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        help='Set random seed. Default value is 42.',
                        required=False, default=42)
    parser.add_argument('--num', type=int,
                        help='Sample num per worker. Default value is 100.',
                        required=False, default=100)
    parser.add_argument('--worker', type=int,
                        help='num of workers. Default value is 1.',
                        required=False, default=1)
    args = parser.parse_args()

    # sampler = SampleGenerator(0, args.worker, args.seed)
    # sampler.ts_sampler_basic(result_path="/data/xiaotannan/ts_data_ieee39", total_num=args.num, cut_length=101)
    workers = [RayWorkerForSampleGenerator.remote(worker_no, args.worker, args.seed) for worker_no in range(args.worker)]
    ray.get([worker.ts_sampler_basic.remote(result_path="/data/xiaotannan/ts_data_ieee39", total_num=args.num, cut_length=101) for worker in workers])

    # worker = SampleGenerator(4242, 1)

    # grid search for sopf
    # grid_sampler_for_sopf([0,8],10,100)

    # d_path = '/home/xiaotannan/pythonPS/00saved_results/models/scopf_agent/final_ddpg_closest_warm_prl/train_and_eval.npz'
    # d = np.load(d_path, allow_pickle=True)
    # data_eval = d['eval'][:11].transpose(1,0)
    # df = pd.DataFrame(data_eval)
    # df.to_excel('data_piece.xlsx', index=False)
    # Picture_Drawer.draw_3d_pic(d[:,0], d[:,1], d[:,4])

    # gen-30, avr-1, ramdom fault, limit angle range, balance stable and unstable
    # worker.ts_sampler_simple_random_for_avr_1(gen_no=0, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/avr_1/gen_30_4000_1000_no_fault_limit_balanced',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=False,
    #                                           check_slack=False,
    #                                           limit_3phase_short=False,
    #                                           must_stable=False,
    #                                           limit_angle_range=True,
    #                                           balance_stability=True,
    #                                           need_larger_than=None,
    #                                           need_smaller_than=None
    #                                           )
    # gen-30, avr-1, limit fault, limit angle range, must stable
    # worker.ts_sampler_simple_random_for_avr_1(gen_no=0, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/avr_1/gen_30_4000_1000_fault_limit_must_stable',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=True,
    #                                           check_slack=True,
    #                                           limit_3phase_short=True,
    #                                           must_stable=True,
    #                                           limit_angle_range=False,
    #                                           balance_stability=False,
    #                                           need_larger_than=None,
    #                                           need_smaller_than=None
    #                                           )

    # gen-33, avr-1, ramdom fault, limit angle range, balance stable and unstable, larger than 3.25, smaller than 0.5
    # worker.ts_sampler_simple_random_for_avr_1(gen_no=3, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/avr_1/gen_33_4000_1000_no_fault_limit_balanced',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=False,
    #                                           check_slack=False,
    #                                           limit_3phase_short=False,
    #                                           must_stable=False,
    #                                           limit_angle_range=True,
    #                                           balance_stability=True,
    #                                           need_larger_than=3.25,
    #                                           need_smaller_than=0.5
    #                                           )
    # gen-33, avr-1, limit fault, limit angle range, must stable, larger than 3.25, smaller than 0.5
    # worker.ts_sampler_simple_random_for_avr_1(gen_no=3, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/avr_1/gen_33_4000_1000_fault_limit_must_stable',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=True,
    #                                           check_slack=True,
    #                                           limit_3phase_short=True,
    #                                           must_stable=True,
    #                                           limit_angle_range=False,
    #                                           balance_stability=False,
    #                                           need_larger_than=3.25,
    #                                           need_smaller_than=0.5
    #                                           )
    
    # gen-31, gen-0, ramdom fault, limit angle range, balance stable and unstable
    # worker.ts_sampler_simple_random_for_gen_0(gen_no=1, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/generator_epie/gen_31_4000_1000_no_fault_limit_balanced',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=False,
    #                                           check_slack=False,
    #                                           limit_3phase_short=False,
    #                                           must_stable=False,
    #                                           limit_angle_range=True,
    #                                           balance_stability=True
    #                                           )
    # gen-31, gen-0, limit fault, limit angle range, must stable
    # worker.ts_sampler_simple_random_for_gen_0(gen_no=1, 
    #                                           result_path='/home/xiaotannan/pythonPS/00saved_results/samples/generator_epie/gen_31_4000_1000_fault_limit_must_stable',
    #                                           num=4000, 
    #                                           test_per=0.2,
    #                                           cut_length=1001, 
    #                                           check_voltage=True,
    #                                           check_slack=True,
    #                                           limit_3phase_short=True,
    #                                           must_stable=True,
    #                                           limit_angle_range=False,
    #                                           balance_stability=False
    #                                           )

    # Bus16-19, district, ramdom fault, limit angle range, balance stable and unstable
    # worker.ts_sampler_simple_random_for_custom_district(#result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_16_19_4000_1000_no_fault_limit_balanced',
    #                                                     result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_solar_16_19_4000_1000_no_fault_limit_balanced',
    #                                                     num=8000, 
    #                                                     test_per=0.1,
    #                                                     cut_length=1001, 
    #                                                     check_voltage=False,
    #                                                     check_slack=False,
    #                                                     limit_3phase_short=False,
    #                                                     must_stable=False,
    #                                                     limit_angle_range=True,
    #                                                     balance_stability=True
    #                                                     )
    # Bus16-19, district, limit fault, limit angle range, must stable
    # worker.ts_sampler_simple_random_for_custom_district(#result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_16_19_4000_1000_fault_limit_must_stable',
    #                                                     result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_with_inner_P_16_19_4000_1000_fault_limit_must_stable',
    #                                                     #result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_solar_16_19_4000_1000_fault_limit_must_stable',
    #                                                     num=8000, 
    #                                                     test_per=0.1,
    #                                                     cut_length=1001, 
    #                                                     check_voltage=True,
    #                                                     check_slack=True,
    #                                                     limit_3phase_short=True,
    #                                                     must_stable=True,
    #                                                     limit_angle_range=False,
    #                                                     balance_stability=False
    #                                                     )

    # Bus16-19, solar district, ramdom fault, limit angle range, balance stable and unstable
    # worker.ts_sampler_simple_random_for_Solar_district(result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_solar_16_19_4000_1000_no_fault_limit_balanced',
    #                                                    num=8000, 
    #                                                    test_per=0.1,
    #                                                    cut_length=1001, 
    #                                                    check_voltage=False,
    #                                                    check_slack=False,
    #                                                    limit_3phase_short=False,
    #                                                    must_stable=False,
    #                                                    limit_angle_range=True,
    #                                                    balance_stability=True
    #                                                    )
    # Bus16-19, solar district, limit fault, limit angle range, must stable
    # worker.ts_sampler_simple_random_for_Solar_district(#result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_solar_16_19_4000_1000_fault_limit_must_stable',
    #                                                    result_path='/home/xiaotannan/pythonPS/00saved_results/samples/district/acline_solar_16_19_limit_short_4000_1000_fault_limit_must_stable',
    #                                                    num_total=8000, 
    #                                                    test_per=0.1,
    #                                                    cut_length=1001, 
    #                                                    check_voltage=True,
    #                                                    check_slack=True,
    #                                                    limit_3phase_short=True,
    #                                                 #    limit_3phase_short=False,
    #                                                    must_stable=True,
    #                                                    limit_angle_range=False,
    #                                                    balance_stability=False
    #                                                    )

    # gen-31, gen-solar, ramdom fault, limit angle range, balance stable and unstable
    # worker.ts_sampler_simple_random_for_gen_9_Solar(gen_no=1, 
    #                                                 result_path='/home/xiaotannan/pythonPS/00saved_results/samples/generator_solar/gen_31_4000_1000_no_fault_limit_balanced',
    #                                                 num=4000, 
    #                                                 test_per=0.2,
    #                                                 cut_length=1001, 
    #                                                 check_voltage=False,
    #                                                 check_slack=False,
    #                                                 limit_3phase_short=False,
    #                                                 must_stable=False,
    #                                                 limit_angle_range=True,
    #                                                 balance_stability=True
    #                                                 )
    # gen-31, gen-solar, limit fault, limit angle range, must stable
    # worker.ts_sampler_simple_random_for_gen_9_Solar(gen_no=1, 
    #                                                 result_path='/home/xiaotannan/pythonPS/00saved_results/samples/generator_solar/gen_31_4000_1000_fault_limit_must_stable',
    #                                                 num=4000, 
    #                                                 test_per=0.2,
    #                                                 cut_length=1001, 
    #                                                 check_voltage=True,
    #                                                 check_slack=True,
    #                                                 limit_3phase_short=True,
    #                                                 must_stable=True,
    #                                                 limit_angle_range=False,
    #                                                 balance_stability=False
    #                                                 )

    # worker.pf_sampler_for_qifeng(num=10000, 
                                #  result_path='/home/xiaotannan/pythonPS/00saved_results/samples/powerflow/ieee39_39x10000')
    # worker.pf_sampler_for_qifeng(num=10000, 
                                #  result_path='/home/xiaotannan/pythonPS/00saved_results/samples/powerflow/2383wp_2383x10000')

    # # worker.ts_sampler_simple_random(10)
    # for i in range(40):
    #     worker.set_seed(i)
    #     worker.simple_random_sampling(0, 25, 0, 50)
    # worker.simple_random_sampling_critical(1, 25, 0, 50)


    """
    worker = RayWorkerForSampleGenerator(0, 1)
    line_table = [['BUS-16', 'BUS-19', 22],
                  ['BUS-21', 'BUS-22', 27],
                  ['BUS-38', 'BUS-29', 110]]
    fault_table = [55]
    worker.all_critical_line_check(line_table=line_table, fault_table=fault_table)
    # """
    # worker = RayWorkerForSampleGenerator(0, 1)
    # worker.simple_random_sampling(0, 1, 0)
    # worker.rigid_grid_search()
    # read_result('../results/trace.DAT')
    # grid_sampler_for_st_gcn()
    worker = SampleGenerator(0, 1)
    worker.ts_sampler_for_lstm(result_path='./00saved_results/samples/lstm_TSA',
                               total_num=10000,
                               round_num=5,
                               cut_length=31)
    """
    se = int(time.time())
    np.random.seed(se)
    test_worker = RayWorkerForSampleGenerator(0, 1)
    results = test_worker.grid_sampler(200)
    ct += time.time()
    print(ct)
    np.savez('./sample.npz', results)
    # """
    """
    total_round = 10
    numWorkers = 40
    ray.init(num_cpus=numWorkers, include_webui=False, ignore_reinit_error=True)
    workers = [RayWorkerForSampleGenerator.remote(i, numWorkers) for i in range(numWorkers)]
    stable_num = unstable_num = 0
    for sample_round in range(total_round):
        cur_result = ray.get([worker.simple_random_sampling.remote(1, 25, sample_round, 50) for worker in workers])
        for result in cur_result:
            stable_num += result[0]
            unstable_num += result[1]
        print('round {} finished, stable vs unstable: {} vs {}'.format(sample_round, stable_num, unstable_num))
    # """
    """
    worker = RayWorkerForSampleGenerator(0, 1)
    for sample_round in range(10):
        worker.grid_sampler_for_st_gcn(10, sample_round)
    # """
    """
    ls = np.load('./sample_1_0.npz', allow_pickle=True)
    x = ls['sample_name']
    x = ls['curve']
    x = ls['label']
    print(ls.shape)
    # """

    # api = psopsAPI()
    # rng = np.random.default_rng(0)
    # samples = list()
    # shape = api.get_load_number()*2
    # cur_central = np.ones(shape)
    # samples.append(cur_central)
    # for i in range(4*24*7):
    #     cur_central -= 0.02
    #     cur_central += rng.random(shape) * 0.04
    #     cur_central[np.where(cur_central < 0.7)] = 0.7
    #     cur_central[np.where(cur_central > 1.2)] = 1.2

    
    # size = 12
    # font1 = {'size': size}
    # mpl.rcParams['xtick.labelsize'] = size
    # mpl.rcParams['ytick.labelsize'] = size
    
    # # training process
    # total_step = 50000
    # interval = 100
    # n_processor = 16
    # training_data = np.load('./00saved_results/models/scopf_agent/shen_20210911/TD3_sopf_0.npz', allow_pickle=True)
    # train = training_data['train'].reshape(-1,n_processor)
    # eval = training_data['eval']
    # x = np.arange(1, total_step+1)
    # plt.xlim((0, total_step))
    # plt.ylim((-1000, 1000))
    # plt.xlabel('Training Step', fontdict=font1)
    # plt.ylabel('Average Reward', fontdict=font1)
    # plt.tick_params(labelsize=size)
    # avg = np.zeros(train.shape[0])
    # for i in range(train.shape[1]): 
    #     avg += train[:, i]
    #     # plt.scatter(x, train[:, i],
    #     #         s = 5,
    #     #         c = 'b',
    #     #         alpha=0.01)
    # avg /= n_processor
    # plt.scatter(x, avg,
    #             s = 5,
    #             label="training", 
    #             c = 'b',
    #             )
    # x = np.arange(0, total_step+1, interval)
    # avg = np.zeros(eval.shape[0])
    # for i in range(eval.shape[1]): avg += eval[:, i]
    # avg /= n_processor * 10
    # plt.plot(x, avg,
    #          label="evaluation", 
    #          color = 'r',
    #          )
    # plt.savefig('./00saved_results/models/scopf_agent/shen_20210911/training.jpg', dpi=300, format='jpg')

    # comparation with PSO
    # n_sample = 1000
    # ylimit = 48000
    # compare_data = np.load('./eval_compare/results.npz', allow_pickle=True)
    # rl = compare_data['rl']
    # pso = compare_data['pso']
    # fail = np.where(rl > 100000.)[0]
    # print(f'{fail.shape[0]} rl control failed')
    # success = np.where(rl < 100000.)[0]
    # diff = ((rl - pso) / pso * 100.)[success]
    # print(f'avg diff {sum(diff) / success.shape[0]}%')
    # print(f'min diff {min(diff)}%')
    # print(f'max diff {max(diff)}%')
    # # plot difference
    # x = np.arange(0, n_sample)
    # plt.xlim((0, n_sample))
    # plt.ylim((0, ylimit))
    # plt.xlabel('Evaluation Sample', fontdict=font1)
    # plt.ylabel('Generation Cost($/h)', fontdict=font1)
    # plt.tick_params(labelsize=size)
    # index = np.argsort(pso)
    # plt.scatter(x, rl[index],
    #             s = 5,
    #             label="proposed method", 
    #             c = 'b',
    #             )
    # plt.scatter(x, pso[index],
    #             s = 5,
    #             label="PSO", 
    #             c = 'r',
    #             )

    # plt.legend(fontsize=size)

    # plot_show()

    ct += time.time()
    print(ct)
    # "Training Step", "Reward",
