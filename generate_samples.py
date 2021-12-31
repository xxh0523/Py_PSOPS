from gym_psops.envs.env_OPF import worker_opf
from sample_generator.sample_generator import SampleGenerator, plot_dots, plot_show
from gym_psops.envs.psops.py_psops import Py_PSOPS
from ray.state import current_node_id
from sample_generator import RayWorkerForSampleGenerator, sample_generator
from sample_generator import read_result
import numpy as np
from multiprocessing.pool import Pool
import ray
import time
import matplotlib as mpl
import matplotlib.pyplot as plt


def grid_sampler_for_st_gcn():
    test_worker = RayWorkerForSampleGenerator(0, 1)
    test_worker.grid_sampler_for_st_gcn()


if __name__ == '__main__':
    ct = -time.time()

    worker = SampleGenerator(0, 1)
    worker.ts_sampler_simple_random_for_gen_0(gen_no=1, num=4000, cut_length=301, limit_angle_range=True,
                                              result_path='./00saved_results/samples/generator_epie/300_gen31_p3_all_4000_masked_samples_gen_0')
    # worker.ts_sampler_simple_random_for_gen_6(gen_no=0, num=4000, cut_length=1001, limit_angle_range=False,
                                            #   result_path='./00saved_results/samples/generator_6/1000_gen30_all_4000_nolimit_samples_gen_6')
    # worker.ts_sampler_simple_random_for_avr_1(gen_no=0, num=4000, cut_length=1001, limit_angle_range=False,
                                            #   result_path='/home/xiaotannan/pythonPS/00saved_results/samples/avr_1/1000_gen30_4000_nolimit_samples_avr_1')

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
