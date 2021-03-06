# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 08:34:02 2020

@author: lance
"""

import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import urllib3
import os
import shutil
from datetime import datetime
from enum import Enum
from numpy.linalg import LinAlgError


# NYT Data
URL_STATES = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv"
NYT_CSV_NAME = "latest.csv"

# The Covid Tracking Project
#   for current.json, use the following fields:
#       date, hospitalizedCurrently, negative, positive, totalTestResults
URL_HISTORIC = "https://covidtracking.com/api/v1/states/daily.json"
URL_CURRENT = "https://covidtracking.com/api/v1/states/current.json"
HISTORIC_JSON_NAME = "daily.json"


IMAGE_DIR = "./images"

def ensure_directory(path: str)-> bool:
    """Attempt to create directory specified by `path'.
    Returns bool: True iff path exists or was succesfully created; otherwise False"""
    def valid(path)-> bool:
        return os.access(path, os.F_OK | os.W_OK)

    if not valid(path):
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            return False
    return valid(path)


class EComparison:
    ABSOLUTE = 1
    PER100K = 2
    PER1M = 3

class Covid:
    DEFAULT_DAYS = [30, 21, 14]
    
    def __init__(self):
        if not ensure_directory(IMAGE_DIR):
            print(f"WARN: Failed to create directory {IMAGE_DIR}")
        
            
        data: pd.DataFrame = self.get_nyt_data()
        daily: pd.DataFrame = self.get_covid_data()
        del daily["state"]
        merged = data.merge(daily, how='inner', left_on=['fips', 'date'], right_on=['fips', 'date'])
        self.data = merged
        self.states = list(self.data.state.unique())
        
    def pull_and_save(self, url: str, out_name: str)-> pd.DataFrame:            
        http = urllib3.PoolManager()
        r = http.request("GET", url)
        with open(out_name, 'wb') as fp:
            fp.write(r.data)
        
        now = datetime.now()
        y, m, d = now.year, now.month, now.day
        suffix = f"_{y:04d}{m:02d}{d:03d}"
        backup = f"{out_name}{suffix}"
        shutil.copy(out_name, backup)
            
    def get_nyt_data(self, url=URL_STATES)-> pd.DataFrame:
        if not os.path.exists(NYT_CSV_NAME):
            self.pull_and_save(url, NYT_CSV_NAME)

        df = pd.read_csv(NYT_CSV_NAME)
        df.date = [datetime.strptime(df.date.iloc[q], "%Y-%m-%d") for q in range(len(df))]
        return df

    def get_covid_data(self, url=URL_HISTORIC)-> pd.DataFrame:
        if not os.path.exists(HISTORIC_JSON_NAME):
            self.pull_and_save(url, HISTORIC_JSON_NAME)
        df = pd.read_json(HISTORIC_JSON_NAME)
        df.date = [datetime.strptime(str(s), "%Y%m%d") for s in df.date] 
        return df
    
        
    def get_list_of_states(self)-> list:
        return list(self.data.state.unique())
    
    def get_state(self, state: str)-> pd.DataFrame:        
        df: pd.DataFrame = self.data[self.data.state == state]
        return df
    
    @staticmethod
    def process_state(data: pd.DataFrame, column_name: str, days: list, dates=None, include_poly=False):
        """computes least square fit on dx/dt[cases] over `days'.
        Returns the  x,y values for each day in `days' and slopes"""
        out = []
        if not column_name is None:
            sel = data[column_name].isna()
            data = data[~sel]
            metric = data[column_name]
        else:
            metric = data
        metric_dev = np.diff(metric)
        metric_dev_a = Covid.running_average(metric_dev, 7)
        
        for day in days:            
            x_int = list(range(day))[:len(metric_dev_a)]
            try:
                poly = np.polyfit(x_int, metric_dev_a[-day:], 1)
            except TypeError as TE:
                #print(f"x_int {len(x_int)} metric_dev_a {len(metric_dev_a[-day:])}")
                if include_poly:
                    out.append(([], [], 0, []))
                else:
                    out.append(([], [], 0))
                continue
            y = np.polyval(poly, x_int)
            m = (y[-1] - y[0]) / day
        
            #if "x" in data:
            #    x = data.date.iloc[-day:]
            #else:
            #    x = dates[-day:]
            
            if not dates is None:
                x = dates[-day:]
            else:
                x = data.date.iloc[-day:]
            if include_poly:
                out.append((x,y,m,poly))
            else:
                out.append((x, y, m))
        return out

    def find_worst_states(self, column_name: str, num_return: int, days: list):
        days.sort()
        all_slopes = []
        
        for state in self.states:
            out = [state]
            data = self.get_state(state)
            try:
                vals = Covid.process_state(data, column_name, days)
            except LinAlgError as LE:
                print(f"ERROR: Failed LstSqr converge for {state}")
                continue
            slopes = [v[-1] for v in vals]
            
            out.append(slopes[0])
            out.append(vals)
            
            all_slopes.append(out)
            all_slopes.sort(key=lambda x:x[1], reverse=True)
            all_slopes = all_slopes[:num_return]
        return all_slopes

    def get_worst_positive(self, num_plot: int):
        temp = []
        for state in self.states:
            data = self.get_state(state)
            tests = data.totalTestResults
            positive = data.positive
            pp = positive / tests
            sel = pp.isna()
            pp[sel] = 0.0
            temp.append((state, data.date, pp, pp.iloc[-1]))
        temp = sorted(temp, key=lambda x:x[3], reverse=True)
        temp = [[x[0], x[1], x[2]] for x in temp]
        return temp[:num_plot]
    
    def find_worst_positive_to_testing(self, num_return: int, day_range: int):
        """Returns states with postive rate higher than testing rate over last `day_range' days.
        Returns: list of list: [state: str, test_increase: float, positive_increase: float, ratio of the two: float]"""
        N = day_range
        vals = []
        for state in self.states:
            s = self.get_state(state)
            ttr = s.totalTestResults.values
            pos = s.positive.values
            test_inc = ttr[-1] / ttr[-N]
            pos_inc = pos[-1] / pos[-N]
            vals.append( (state, test_inc, pos_inc, pos_inc / test_inc) )
        vals_sorted = sorted(vals, reverse=True, key=lambda x: x[-1])
        worst = list(filter( lambda x: x[-1] > 1, vals_sorted)) 
        return worst[:num_return]

    def print_worst_positive_to_testing(self, worst):
        lens = [len(w[0]) for w in worst]
        w = max(lens)
        fmt = "%%-%ds" % (w)
        print("%s PosInc TestInc Ratio" % (fmt % "State"))
        for w in worst:
            print("%-s %5d%% %5d%% %4.1fx" % (fmt % w[0], w[2]*100, w[1]*100, w[3]))

    def plot_worst(self, worst, metric_name:str):
        for w in worst:
            state = w[0]
            trends = w[2]
            self.plot_state(state, metric_name, trends)
            
    def plot_state(self, state: str, metric_name: str, trends=None)-> plt.Figure:
        data = self.get_state(state)
        metric = data[metric_name]
        metric = data[metric_name].fillna(0)
        metric_dev = np.diff(metric)
        metric_dev_a = Covid.running_average(metric_dev, 7)
        
        fig, ax = plt.subplots(1,2)
        fig.set_tight_layout(True)
        fig.set_figwidth(15)
        fig.set_figheight(6)
        
        ax[0].set_title(f"{state} - {metric_name}")
        ax[0].grid()
        ax[0].set_xlabel("Date")
        ax[0].set_ylabel(f"{metric_name}")
        ax[0].plot(data.date, metric, 'b-', label=metric_name, linewidth=3)
        ax[0].legend(loc="upper left")
        
        ax2 = ax[0].twinx()
        ax2.plot(data.date.iloc[1:], metric_dev, 'r--', alpha=0.5, label='Daily Change')
        ax2.plot(data.date.iloc[1:], metric_dev_a, 'g--', label='Running Average Change', linewidth=2)
        ax2.set_ylabel("Daily")
        
        if not trends:
            trends = Covid.process_state(data, metric_name, Covid.DEFAULT_DAYS)
            
        for day in trends:
            ax2.plot(day[0], day[1], 'k--', label=f"Trend {len(day[0])}-Day")
        ax2.legend(loc="center left")


        # subplot for trends
        ax = ax[1]
        ax.plot(data.date, data.positive.astype(np.float) / data.totalTestResults * 100, 'rx-', label='Percent Positive')
        ax.set_ylabel('Percent Positive', color='r')
        ax.legend(loc='upper left')
        ax.set_ylim((0, 15))
        ax.grid()
        ax2 = ax.twinx()
        ax2.plot(data.date, data.totalTestResults, label='Total Tests')
        ax2.set_ylabel('Total Tests')
        ax2.legend(loc='lower left')

        fig.autofmt_xdate()
        
        return fig

    def get_slopes_of_state(self, state:str, metric_name: str, num_samples: int, days: list):
        """Returns least square regression slopes.
        num_samples: number of samplesst to return, where the offset will be min( `days' )
        days: list of number of days for the trends. e.g. [14, 21, 30]"""
        slopes = []
        
        days = np.array(sorted(days, reverse=True))
        offset = min(days) # (num_samples - 1) * days[0]
        days += offset

        data = self.get_state(state)
        metric = data[metric_name]
        metric = Covid.running_average(np.diff(metric), 7)

        for samp_num in range(num_samples, 0, -1):
            for day in days:
                N = day
                if samp_num > 1:
                    N -= offset

                x_int = range(N)

                poly = np.polyfit(x_int, metric[-N:][:N], 1)
                y = np.polyval(poly, x_int)

                m = (y[-1] - y[0]) / day
                slopes.append(m)
            days -= offset
        return slopes

    @staticmethod
    def get_slopes(metric, num_samples: int, days: list):
        if not isinstance(metric, np.ndarray):
            metric = metric.values
        days.sort(reverse=True)
        days = np.array(days)

        slopes = []
        
        for multipler in range(num_samples, 0, -1):
            for _day in days:
                day_0 = -(_day * multipler)
                day_n = day_0 +_day

                # correct zero value for correct numpy backwards indexing
                if day_n == 0:
                    day_n = -1
                print(f"_day {_day} day_0 {day_0} day_n {day_n}")

                m = (metric[day_n] - metric[day_0]) / _day
                slopes.append(( - day_0, m))
        return slopes


    def plot_all_trends(self, metric_name: str, num=60):
        title = f"{metric_name} - Rate Increases over {Covid.DEFAULT_DAYS} days.\nRecent Period in Blue and prior period in Yellow"
        cols = int(np.ceil(num ** 0.5))
        rows = num // cols
        fig, ax = plt.subplots(rows, cols, sharex=True, sharey=True)
        fig.suptitle(title)
        fig.set_figwidth(11)
        fig.set_figheight(6)
        worst = self.find_worst_states(metric_name, num, Covid.DEFAULT_DAYS)
        states = [w[0] for w in worst]
        del worst
        
        ind = np.arange(3)
        width = 0.45

        all_slopes = []

        for i, state in enumerate(states):
            row = i // cols
            col = i % cols
            slopes = self.get_slopes_of_state(state, metric_name, 2, Covid.DEFAULT_DAYS)
            all_slopes.append(slopes)
            
            ax[row, col].bar(ind, slopes[:3], width=width, color='y')
            ax[row, col].bar(ind + width, slopes[3:], width=width, color='b')
            ax[row, col].grid()
            ax[row, col].set_title(state)        
        return all_slopes, states

    def get_all_by_date(self, go_back_days=None):
        data = self.data.copy()

        dates = data.date.unique()
        out = {m : [] for m in ["cases", "deaths"]}
        
        for date in dates:
            temp = data[data.date == date]
            cases_sum = temp.cases.sum()
            deaths_sum = temp.deaths.sum()
            
            #ny = temp[temp.state=="New York"]
            #if ny.size > 0:
            #    cases_sum -= ny.cases.values[0]
            #    deaths_sum -= ny.deaths.values[0]

            out["cases"].append(cases_sum)
            out["deaths"].append(deaths_sum)

        if not go_back_days is None:
            dates = dates[:-go_back_days]
            for key in out.keys():
                out[key] = out[key][:-go_back_days]
        return dates, out
        
        
    @staticmethod
    def running_average(data, days:int):
        h = days // 2
        out = np.array(data)
        for i in range(h, len(data)-h):
            a = i - h
            b = a + days
            temp = out[a: b].mean()
            out[i] = temp
        return out

    @staticmethod
    def get_first_derivative(data, days:int):
        dev = np.diff(data)
        out = Covid.running_average(dev, days)
        return out

    def plot_all_by_date(self, fig=None, limits=None, go_back_days=None, max_tuple=None):
        dates, db = self.get_all_by_date(go_back_days)
        keys = list(db.keys())
        #if not go_back_days is None:
        #    dates = dates[:-go_back_days]
        #    for key in keys:
        #        db[key] = db[key][:-go_back_days]

        num_keys = len(keys)
        
        reuse = False
        if fig is None:
            fig, ax = plt.subplots(1, num_keys, sharex=True)
            fig.set_size_inches(19.1, 9.0)
        else:
            ax = fig.axes
            reuse = True
            for idx, lim in enumerate(limits):
                fig.axes[idx].set_xlim(lim[0])
                fig.axes[idx].set_ylim(lim[1])

        most_recent_date = np.datetime_as_string(dates[-1], unit='D')
        for i, key in enumerate(keys):
            if len(ax[i].lines) > 0:
                [ax[i].lines[0].remove() for _ in range(len(ax[i].lines))]

            ax[i].plot(dates, db[key], 'b-', label=key)
            ax[i].set_title(f"{key} - {most_recent_date}")
            if not reuse:
                ax[i].grid()
            ax[i].set_ylabel(key)

            if not max_tuple is None:
                ax[i].plot(max_tuple[0], max_tuple[i + 1], c='r', ms=10, marker='o', fillstyle='none', label='Current Max')
                pass
    
            days_arr = sorted(Covid.DEFAULT_DAYS, reverse=True)            
            trends = Covid.process_state(db[key], None, days_arr, dates, include_poly=True)
            
            lsarr = ['--', '-.', ':']
            for idx, day in enumerate(trends):
                poly = day[-1]
                pred_x, pred_y= Covid.predict(day[0], 30, poly)
                pred_y = np.cumsum(pred_y) + db[key][-1]
                ax[i].plot(pred_x, pred_y, color='b', ls=lsarr[idx], label=f"Predict-{len(day[0])}")
            ax[i].legend(loc="upper left")
            
            if reuse:
                ax2 = ax[i + 2]
                ax2.cla()
            else:
                ax2 = ax[i].twinx()
            ax2.set_ylabel("Daily Change")
            ax2.plot(dates[1:], Covid.get_first_derivative(db[key], 7), label="Daily Change")
            for day in trends:
                ax2.plot(day[0][:len(day[1])], day[1], 'k--', label=f"Trend {len(day[0])}-Day")
            ax2.legend(loc="lower right")
        if not reuse:
            limits = [(x.get_xlim(), x.get_ylim()) for x in fig.axes]
        else:
            for idx, lim in enumerate(limits):
                fig.axes[idx].set_xlim(lim[0])
                fig.axes[idx].set_ylim(lim[1])

        return fig, limits, dates, db

    
    def animate_all_by_date(self, days:list):

        days = sorted(days, reverse=True)

        # get all current data
        fig, limits, dates, db = self.plot_all_by_date()
        limits = [(x, (y[0], int(np.ceil(y[1]/2000))*2000)) for (x,y) in limits]
        max_cases = db["cases"][-1]
        max_deaths = db["deaths"][-1]
        max_tuple = (dates[-1], max_cases, max_deaths)
        print(max_tuple)

        for idx, lim in enumerate(limits):
            fig.axes[idx].set_xlim(lim[0])
            fig.axes[idx].set_ylim(lim[1])


        for idx, day in enumerate(days):
            print (f"Working index {idx} with day {day}")
            _fig, _limits, _dates, _db = self.plot_all_by_date(fig, limits, go_back_days=day, max_tuple=max_tuple)
            outname = f"cases_deaths_{idx:03d}_{day:03d}_days_back.png"
            _fig.savefig(create_img_dir_path(outname))
            #plt.close(_fig)
        day = 0
        idx += 1
        _fig, _limits, _dates, _db = self.plot_all_by_date(fig, limits, max_tuple=max_tuple)
        outname = f"cases_deaths_{idx:03d}_{day:03d}_days_back.png"
        _fig.savefig(create_img_dir_path(outname))
        plt.close(_fig)
        

    @staticmethod
    def predict(x, num_days, poly):
        new_x = [x[-1] + np.timedelta64(i, "D") for i in range(1, num_days)]
        new_y = np.polyval(poly, range(num_days))
        return new_x, new_y[:len(new_x)]

def create_img_dir_path(image_name: str)-> str:
    return os.path.join(IMAGE_DIR, image_name)
        
if __name__ == '__main__':
    import sys
    animate = False
    if len(sys.argv) >= 2:
        animate=True
        print("Animating...")
    plt.close('all')
    plt.ioff()
    c = Covid()

    if animate:
        days = range(4, 90, 4)
        c.animate_all_by_date(days)
    
    slopes_cases, states_cases = c.plot_all_trends("cases", 16)
    slopes_deaths, states_deaths =c.plot_all_trends("deaths", 16)

    common = set(states_cases).intersection(set(states_deaths))
    print("common", common)

    fig, limits, dates, db = c.plot_all_by_date()
    plt.show()
    
    if False:
        print("Cases")
        worst_cases = c.find_worst_states("cases", 10, Covid.DEFAULT_DAYS)
        print([w[0] for w in worst_cases])
        c.plot_worst(worst_cases, "cases")
        plt.show()
        
        
        print ("\nDeaths")
        worst_deaths = c.find_worst_states("deaths", 10, Covid.DEFAULT_DAYS)
        print([w[0] for w in worst_deaths])    
        c.plot_worst(worst_deaths, "deaths")
        plt.show()
            

        slopes = c.get_slopes_of_state("Ohio", "cases", 2, Covid.DEFAULT_DAYS)
        c.plot_state("Ohio", "cases")


        plt.show()
            
                
