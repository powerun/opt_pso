
import os
import zipfile
import pymysql
import logging
import datetime
import pandas as pd
import configparser
from sqlalchemy import create_engine
from dbutils.pooled_db import PooledDB
from logging.handlers import TimedRotatingFileHandler


class DataMysql(object):
    def __init__(self, db_conf):
        config = configparser.ConfigParser()
        config.read(db_conf)
        self.host = config['DATABASE']['host']
        self.user = config['DATABASE']['user']
        self.pwd = config['DATABASE']['password']
        self.port_num = int(config['DATABASE']['port'])
        self.db_name = config['DATABASE']['db']
        self.charset = config['DATABASE']['charset']

    def pool_open(self, log):
        try:
            pool = PooledDB(pymysql, 10, host=self.host, user=self.user, password=self.pwd, database=self.db_name)
        except Exception as e:
            log.logger.warning(e)
        else:
            return pool

    def sql_open(self, log):
        try:
            opens = pymysql.connect(host=self.host, user=self.user, password=self.pwd, database=self.db_name)
        except Exception as e:
            log.logger.warning(e)
        else:
            return opens


class Logger(object):
    def __init__(self, filename, fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(sh)
        # fh = logging.FileHandler(filename, encoding='utf-8')
        # fh.setFormatter(logging.Formatter(fmt))
        # self.logger.addHandler(fh)
        trh = TimedRotatingFileHandler(filename=filename, when="D", interval=1, backupCount=7, encoding='utf-8')
        trh.setFormatter(logging.Formatter(fmt))
        self.logger.addHandler(trh)


def exe_time(func):
    def new_func(*args, **args2):
        t0 = datetime.datetime.now()
        data = func(*args, **args2)
        t1 = datetime.datetime.now()
        print(func.__name__, t1 - t0)
        return data
    return new_func


class OptTemp(object):

    def __init__(self, log):
        self.log = log
        self.sql_open = None

    def set_param(self, db_conf='config.opt'):
        config = configparser.ConfigParser()
        config.read(db_conf)

    def open_mysql(self, db_conf):
        sql = DataMysql(db_conf)
        self.sql_open = sql.pool_open(self.log)

    def close_mysql(self):
        if self.sql_open:
            self.sql_open.close()
            self.sql_open = None

    def write_data(self, dat, sql_into):
        conn = self.sql_open.connection()
        cur = conn.cursor()
        try:
            if isinstance(dat, dict):
                for k, v in dat.items():
                    # self.log.logger.info(k)
                    cur.execute(sql_into.format((k,) + v))
            else:
                for i in dat:
                    # self.log.logger.info(i)
                    cur.execute(sql_into.format(tuple(i)))
            conn.commit()
        except Exception as e:
            self.log.logger.warning(e)
            conn.rollback()
        finally:
            cur.close()
            conn.close()

    def query_data(self, query_info):
        conn = self.sql_open.connection()
        try:
            dat_data = pd.read_sql(query_info, conn)
        except Exception as e:
            self.log.logger.warning(e)
        else:
            return dat_data
        finally:
            conn.close()

    def exec_query(self, sql_exec):
        conn = self.sql_open.connection()
        cur = conn.cursor()
        try:
            cur.execute(sql_exec)
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.log.logger.warning(e)
        finally:
            cur.close()
            conn.close()

    def input_data(self, file_name, file_dir='opt_data'):
        z = zipfile.ZipFile(file_dir + '/' + '%s.zip' % file_name)
        z_list = z.namelist()
        for name in z_list:
            z.extract(name, file_dir)
        z.close()
        os.rename(os.path.join(file_dir, z_list[0]), os.path.join(file_dir, z_list[0].encode('cp437').decode('gbk')))
        config = configparser.ConfigParser()
        config.read('config.opt')
        host = config['DATABASE']['host']
        user = config['DATABASE']['user']
        pwd = config['DATABASE']['password']
        db_name = config['DATABASE']['db']
        col_dict = {
            'cop_curve': ['TYPE', 'TEO', 'TCI', 'PAYLOAD', 'COP'],
            'm_types': ['type', 'm_nums', 'rate_cold', 'test_min_p', 'assist_kw', 'assist_alpha', 'approach', 'min_tci', 'max_tci'],
            'time_need_tweb': ['hours', 'tweb', 'cold_need', 'teo']
        }
        engine = create_engine('mysql+pymysql://%s:%s@%s:3306/%s?charset=utf8' % (user, pwd, host, db_name))
        now_time = datetime.datetime.now()
        for root, dirs, files in os.walk(file_dir):
            for f_name in files:
                prj = f_name.split('.')
                if 'xls' in prj[1]:
                    try:
                        engine.execute("INSERT INTO project_id(ProjectName, CreateTime) VALUES ('%s', '%s')" % (prj[0], now_time))
                        dat_excel = pd.ExcelFile(os.path.join(root, f_name))
                        for k in dat_excel.sheet_names:
                            dat = dat_excel.parse(k)
                            dat.columns = col_dict[k]
                            dat['ID'] = prj[0]
                            dat['CreateTime'] = now_time
                            engine.execute(" DELETE FROM %s WHERE ID='%s' " % (k, prj[0]))
                            dat.to_sql(k, engine, index=False, if_exists='append')
                        dat_excel.close()
                    except Exception as e:
                        self.log.logger.warning(e)
                    else:
                        os.remove(os.path.join(root, f_name))
            if not os.listdir(root):
                os.rmdir(root)


















