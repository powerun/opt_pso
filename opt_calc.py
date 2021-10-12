
import os
import opt_pso
import mysql_db
import datetime
import opt_report
import numpy as np
import pandas as pd
import multiprocessing
from functools import reduce
from scipy import interpolate


class OptOpenState(object):
    def __init__(self, log):
        self.logs = log
        self.df_cop = None
        self.df_types = None
        self.df_cold = None
        self.state_list = None
        self.types_list = None
        self.state_load = None

    @staticmethod
    def lists_combination(lists, code=''):
        def my_func(list1, list2):
            return [str(i) + code + str(j) for i in list1 for j in list2]
        return reduce(my_func, lists)

    def set_param(self, df_types, df_cop, df_cold):
        self.df_types = df_types
        self.df_cop = df_cop
        self.df_cold = df_cold
        state_list = self.lists_combination([list(range(i + 1)) for i in self.df_types["m_nums"].tolist()])
        self.state_list = state_list[1:]
        self.types_list = self.df_types['type'].tolist()
        self.state_load = self.manual_operate()

    def opt_key(self, cold_dat, add_cold, tci, teo, split_num=0, dot_tci=0, dot_teo=0):
        tci = round(tci, dot_tci)
        teo = round(teo, dot_teo)
        try:
            cold_dat['tci'] = np.clip(cold_dat['tweb'] + self.df_types.at[0, 'approach'], self.df_types.at[0, 'min_tci'], self.df_types.at[0, 'max_tci'])
            if 0 < split_num <= 50:
                split_num = int(100/split_num)
                cold_dat['cold_need2'] = (cold_dat['cold_need'] / split_num).astype('int') * split_num + split_num / 2
            else:
                cold_dat['cold_need2'] = cold_dat['cold_need']
            cold_dat = cold_dat.round({'tci': dot_tci, 'teo': dot_teo, 'cold_need2': 0})
            cold_dat['key'] = cold_dat['tci'].astype('str') + '_' + cold_dat['teo'].astype('str') + '_' + cold_dat['cold_need2'].astype('str')
            key_dat = cold_dat[cold_dat['cold_need'] > 0]['key'].unique()
            df_x = cold_dat[cold_dat['cold_need'] > 0][['tweb', 'teo', 'tci']]
        except Exception as e:
            self.logs.logger.warning(e)
        else:
            key_dat = (tuple(map(float, i.split('_'))) for i in key_dat)
            key_dat2 = ((tci, teo, i) for i in add_cold)
            key_sum = tuple(key_dat) + tuple(key_dat2)
            df_x.columns = ['tweb', 'TEO', 'TCI']
            return cold_dat, key_sum, df_x

    def cop_curve(self, df_interp, key_sum):
        curve_sum = {}
        for tci_j, teo_j, _ in key_sum:
            curve_key = '_'.join([str(tci_j), str(teo_j)])
            tag = curve_sum.get(curve_key, 0)
            if tag == 0:
                cop_curve_dict = {}
                try:
                    for type_i in self.types_list:
                        df = df_interp[df_interp['TYPE'] == type_i]
                        df = df[(df['TEO'] == teo_j) & (df['TCI'] == tci_j)].reset_index(drop=True)
                        if len(df) > 3:
                            cop_curve_dict[type_i + 'curve'] = interpolate.interp1d(df['PAYLOAD'], df['COP'], kind="quadratic")
                        else:
                            self.logs.logger.info(curve_key)
                except Exception as e:
                    self.logs.logger.warning(e)
                curve_sum.update({curve_key: cop_curve_dict})
        return curve_sum

    def cop_interp(self, df_cop, c1, c2, x, y, df_x):
        df_interp = pd.DataFrame()
        try:
            for type_i in self.types_list:
                df = df_cop[df_cop['TYPE'] == type_i]
                c1_list = df[c1].unique().tolist()
                c2_list = df[c2].unique().tolist()
                for c1_i in c1_list:
                    for c2_i in c2_list:
                        df_i = df[(df[c1] == c1_i) & (df[c2] == c2_i)]
                        if df_x.empty:
                            x_new = list(range(self.df_types.at[0, 'test_min_p'], 101, 5))
                            x_new[-1] = 100
                        else:
                            x_new = df_x[df_x[c1] == c1_i][x].unique().tolist()
                            if x == 'TCI':
                                x_new_min = self.df_types.at[0, 'min_tci']
                                x_new_max = self.df_types.at[0, 'max_tci']
                            else:
                                x_new_min = df_i[x].min()
                                x_new_max = df_i[x].max()
                            x_new = np.unique(np.clip(x_new, x_new_min, x_new_max))
                        if len(df_i) > 3:
                            interp_fun = interpolate.UnivariateSpline(df_i[x], df_i[y], s=0.0035, k=2)
                            y_new = interp_fun(x_new)
                            df_i = pd.DataFrame(np.array([[c1_i]*len(x_new), [c2_i]*len(x_new), x_new, y_new]).T, columns=[c1, c2, x, y], dtype=float)
                            df_i['TYPE'] = [type_i]*len(x_new)
                            df_interp = df_interp.append(df_i)
                        else:
                            self.logs.logger.info('Below three')
                            return df_cop
        except Exception as e:
            self.logs.logger.warning(e)
        return df_interp

    def objective(self, p_list, **kwargs):  # p_list表示各台机器的加载率
        main_power = 0
        assist_power = 0
        for i, type_i in enumerate(self.types_list):
            cop = kwargs.get(type_i + 'curve')([p_list[i]])
            type_num = kwargs.get(type_i)
            main_power += type_num * self.df_types.at[i, 'rate_cold'] * 3.517 * p_list[i]/100 / cop[0]
            assist_power += self.df_types.at[i, 'assist_kw'] * (p_list[i]/100) ** self.df_types.at[i, 'assist_alpha']
        return main_power+assist_power

    def constraints(self, p_list, **kwargs):
        cold = kwargs.get('cold', 0)
        supply = 0
        for i, type_i in enumerate(self.types_list):
            type_num = kwargs.get(type_i)
            supply += type_num * p_list[i]/100 * self.df_types.at[i, 'rate_cold']
        supply_more = supply - cold   # 供需平衡约束
        return [supply_more]

    def manual_operate(self):
        state_load = np.array([])
        for state_i in self.state_list:
            state_max_load = 0
            for i, type_i in enumerate(self.types_list):
                state_max_load += int(state_i[i]) * self.df_types.at[i, 'rate_cold']
            state_load = np.append(state_load, state_max_load)
        return state_load

    def opt_result(self, pr, cold_dat, multi_dict, table_in):
        result = []
        compute_col = ['min', 'people', 'min_index', 'peo_index', 'cop_min', 'cop_peo']
        state_col = ['openState_' + i for i in self.state_list]
        key_col = ['tci', 'teo', 'cold']
        try:
            res_key = np.where(cold_dat['cold_need'] > 0, cold_dat['key'], 0)
            for row in res_key:
                if row == 0:
                    result.append([0] * (len(key_col) + len(self.state_list) + len(compute_col)))
                else:
                    res_2 = multi_dict.get(row, [[], []])
                    res = res_2[0]
                    res_power_all = res[3:]
                    if res_power_all:
                        res_power = list(filter(lambda x: x < 10000, res_power_all))
                        if res_power:
                            res_min = min(res_power)
                            res_min_index = res_power_all.index(res_min)
                            res_peo = res_2[1][0]
                            res_peo_index = res_2[1][1]
                            cop_min = 3.517 * res[2] / res_min
                            cop_peo = 3.517 * res[2] / res_peo
                            res = res + [res_min, res_peo, res_min_index, res_peo_index, cop_min, cop_peo]
                        else:
                            res = res + [0] * len(compute_col)
                        result.append(res)
                    else:
                        self.logs.logger.info(row)
        except Exception as e:
            self.logs.logger.warning(e)
        col = key_col + state_col + compute_col
        result = pd.DataFrame(result, columns=col)
        result['index'] = result.index + 1
        calc_insert = result[['index'] + compute_col]
        calc_insert = calc_insert[calc_insert['min'] > 0]
        sql_insert_sum = []
        res_insert_sum = []
        for i in self.state_list:
            state_i = 'openState_' + i
            res_select = result[key_col + ['index', state_i]]
            res_select = res_select[(res_select[state_i] < 10000) & (res_select[state_i] > 0)]
            sql_insert = ''' INSERT INTO %s(ID, tci, teo, cold, hours, min_value, state, CreateTime)
                                     VALUES ('%s','{0[0]}','{0[1]}','{0[2]}','{0[3]}','{0[4]}', '%s',now())
                                     ''' % (table_in, pr, i)
            res_insert_sum.append(res_select)
            sql_insert_sum.append(sql_insert)
        return res_insert_sum, sql_insert_sum, calc_insert

    def opt_main(self, tci, teo, cold, curve_sum, multi_dict, lock):
        rows = [tci, teo, cold]
        peo_power, peo_index, min_value = 0, 0, 0
        my_dict = curve_sum.get('_'.join([str(tci), str(teo)]), {})
        try:
            for state_i in self.state_list:
                state_min_load, state_max_load = 0, 0
                for i, type_i in enumerate(self.types_list):
                    type_num = int(state_i[i])
                    state_min_load += type_num * self.df_types.at[i, 'rate_cold'] * self.df_types.at[i, 'test_min_p'] / 100
                    state_max_load += type_num * self.df_types.at[i, 'rate_cold']
                    my_dict.update({type_i: type_num})
                my_dict['cold'] = cold
                if cold <= state_min_load:
                    min_value = 10000
                elif cold >= state_max_load:
                    if state_i == self.state_list[-1]:
                        min_value = self.objective([100, 100], **my_dict)
                    else:
                        min_value = 10001
                else:
                    lb = self.df_types['test_min_p'].tolist()
                    ub = [100] * len(lb)
                    opt_res = opt_pso.PSO(self.objective, lb, ub, f_ieqcons=self.constraints, kwargs=my_dict, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=50)
                    min_value = opt_res.pso()[1]
                if min_value == 10000:
                    rows = rows + [min_value] * (len(self.state_list) - len(rows[3:]))
                    break
                else:
                    rows.append(min_value)
            if np.sum(np.array(rows) < 10000):
                peo_index = np.sum(self.state_load < cold)
                peo_index = np.clip(peo_index, 0, len(self.state_load)-1)
                p_list = cold / self.state_load[peo_index] * 100
                p_list = np.clip(p_list, self.df_types.at[0, 'test_min_p'], 100)
                for i, type_i in enumerate(self.types_list):
                    type_num = int(self.state_list[peo_index][i])
                    my_dict.update({type_i: type_num})
                peo_power = self.objective([p_list, p_list], **my_dict)
        except Exception as e:
            self.logs.logger.warning(e)
        lock.acquire()
        multi_dict.update({'_'.join([str(tci), str(teo), str(cold)]): [rows, [peo_power, peo_index]]})
        lock.release()


def multi_main(report_folder, tci=31, teo=6):
    logs = mysql_db.Logger('./logs/' + str(datetime.datetime.now().date()))
    my_sql = mysql_db.OptTemp(logs)
    my_sql.open_mysql('config.opt')
    test = OptOpenState(logs)
    cpu_num = os.cpu_count()
    tci = float(tci)
    teo = float(teo)
    add_cold = [166.7, 250.0, 333.3, 416.7, 500.0, 583.3, 666.7, 750.0, 833.3,
                916.7, 1000.0, 1083.3, 1166.7, 1250.0, 1333.3, 1416.7, 1500.0,
                1583.3, 1666.7, 1750.0, 1833.3, 1916.7, 2000.0]
    cold_plot = pd.DataFrame(add_cold, columns=['cold_need'])
    my_sql.input_data(report_folder)
    cold_plot['key'] = str(tci) + '_' + str(teo) + '_' + cold_plot['cold_need'].astype('str')
    prj = my_sql.query_data(''' SELECT ProjectName FROM project_id WHERE CreateTime > '%s' ''' % report_folder)
    prj = prj['ProjectName'].values
    for pr in prj:
        df_cop = my_sql.query_data(''' SELECT * FROM cop_curve WHERE ID='%s' ''' % pr)
        df_types = my_sql.query_data(''' SELECT * FROM m_types WHERE ID='%s' ''' % pr)
        df_cold = my_sql.query_data(''' SELECT * FROM time_need_tweb WHERE ID='%s' ''' % pr)
        my_sql.exec_query(''' DELETE FROM opt_value WHERE ID='%s' ''' % pr)
        my_sql.exec_query(''' DELETE FROM opt_plot WHERE ID='%s' ''' % pr)
        my_sql.exec_query(''' DELETE FROM opt_index WHERE ID='%s' ''' % pr)
        test.set_param(df_types, df_cop, df_cold)
        cold_dat, key_sum, df_x = test.opt_key(df_cold, add_cold, tci, teo)
        df_interp = test.cop_interp(df_cop, c1='TCI', c2='TEO', x='PAYLOAD', y='COP', df_x=pd.DataFrame())  # 对加载率插值
        df_interp = test.cop_interp(df_interp, c1='TCI', c2='PAYLOAD', x='TEO', y='COP', df_x=df_x)  # 对冷冻水出水温度插值
        df_interp = test.cop_interp(df_interp, c1='TEO', c2='PAYLOAD', x='TCI', y='COP', df_x=df_x)  # 对冷却水进水温度插值
        curve_sum = test.cop_curve(df_interp, key_sum)
        if curve_sum:
            try:
                pool = multiprocessing.Pool(processes=cpu_num)
                multi_dict = multiprocessing.Manager().dict()
                lock = multiprocessing.Manager().Lock()
                for tci_i, teo_i, cold_i in key_sum:
                    pool.apply_async(test.opt_main, (tci_i, teo_i, cold_i, curve_sum, multi_dict, lock,))
                pool.close()
                pool.join()
                dat_in, sql_in, calc_in = test.opt_result(pr, cold_dat, multi_dict, 'opt_value')
                dat_in2, sql_in2, _ = test.opt_result(pr, cold_plot, multi_dict, 'opt_plot')
                for i, j in zip(dat_in, sql_in):
                    my_sql.write_data(i.values, j)
                for i, j in zip(dat_in2, sql_in2):
                    my_sql.write_data(i.values, j)
                sql_insert_calc = ''' INSERT INTO %s(ID, hours, min_value, peo_value, min_index, peo_index, min_cop, peo_cop, CreateTime)
                                      VALUES ('%s','{0[0]}','{0[1]}','{0[2]}','{0[3]}','{0[4]}', '{0[5]}','{0[6]}',now())
                                      ''' % ('opt_index', pr)
                my_sql.write_data(calc_in.values, sql_insert_calc)
            except Exception as e:
                logs.logger.warning(e)
            else:
                logs.logger.info(pr+' success')
    my_sql.close_mysql()
    try:
        opt_report.OptReport(prj).report_main(report_folder)
    except Exception as e:
        logs.logger.warning(e)







