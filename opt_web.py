
from flask import Flask, render_template, request, send_from_directory
import os
import datetime
import psutil
import configparser
import opt_calc
from multiprocessing import Process
app = Flask(__name__)


@app.route("/opt/", methods=['GET', 'POST'])
def opt_html():
    if request.method == 'POST':
        report_folder = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H-%M-%S')
        f_one = request.files['file_one']
        f_one.save('./opt_data/%s.zip' % report_folder)
        f_two = request.form.get('file_two')
        f_two = f_two.split('-')
        if len(f_two) == 2:
            p = Process(target=opt_calc.multi_main, args=(report_folder, int(f_two[0]), int(f_two[1]), ))
            p.start()
            time_tag = '_'.join([report_folder.replace(' ', '_'), str(p.pid)])
            return render_template('index2.html', time_tag=time_tag)
        else:
            return "<h1>fail</h1>"
    else:
        return render_template('index.html')


@app.route("/opt/example/", methods=['GET'])
def download_example():
    return send_from_directory("./templates", path="example.zip", as_attachment=True)


@app.route("/opt/res/<num>", methods=['GET'])
def download_wait(num):
    num = num.split('_')
    rep_zip = ' '.join(num[:-1])
    end_time = datetime.datetime.now()
    start_time = datetime.datetime.strptime(rep_zip, "%Y-%m-%d %H-%M-%S")
    cost_time = round((end_time - start_time).seconds / 60, 1)
    try:
        p = psutil.Process(int(num[-1]))
    except Exception as e:
        if os.path.isfile('./opt_report/%s.zip' % rep_zip):
            return render_template('index4.html', result_zip=rep_zip+'.zip', time_tag=cost_time, download='_'.join(num[:-1]))
        else:
            return render_template('index5.html', time_tag=cost_time)
    else:
        return render_template('index3.html', time_tag=cost_time)


@app.route("/opt/res/1/<num>", methods=['GET'])
def download_result(num):
    zip_path = '%s.zip' % num.replace('_', ' ')
    return send_from_directory("./opt_report", path=zip_path, as_attachment=True)


def get_host_port(db_config='config.opt'):
    config = configparser.ConfigParser()
    config.read(db_config)
    host = config['APPRUN']['host']
    post = int(config['APPRUN']['port'])
    return host, post


if __name__ == "__main__":
    host_i, port_i = get_host_port()
    app.run(host=host_i, port=port_i, debug=False)



