# Parsing the csv to create training/evluation dataset.
import csv
from datetime import datetime
from params import Params as param
import numpy as np

class STRdataloader():

    def __init__(self):
        """
        Args:
            path (str): string path to a csv file.
            param (Params): params for configuring training.
        """
        self.csv_reader = csv.DictReader(open(param.data_path, "r")) 
        # Each frame presenets the accidental statistic of a single day
        # Embeded in a 2D     
        self.frames = []
        self.param = param
        self.height = param.map_height
        self.width = param.map_width

        # Step 1: iterate csv dataset to calulate days and shape..
        self.date_min = None
        self.date_max = None
        self.x_max = None
        self.x_min = None
        self.y_max = None
        self.y_min = None 
        for row in self.csv_reader:
            date = self.row_to_datetime(row) 
            x, y = self.row_to_coordinates(row)

            if x is None or y is None:
                continue

            if self.date_max is None or date > self.date_max:
                self.date_max = date
            if self.date_min is None or date < self.date_min:
                self.date_min = date

            if self.x_min is None or x < self.x_min:
                self.x_min = x
            if self.x_max is None or x > self.x_max:
                self.x_max = x

            if self.y_min is None or y < self.y_min:
                self.y_min = y

            if self.y_max is None or y > self.y_max:
                self.y_max = y

        self.num_days = (self.date_max - self.date_min).days + 1
        self.frames = [[[0 for i in range(self.width)] for j in range (self.height)] for k in range(self.num_days)]
        self.grid_width = (self.x_max - self.x_min + 0.0000001) / self.width
        self.grid_height = (self.y_max - self.y_min + 0.0000001) / self.height
        # Step 2: parse the csv file to conver it into grids
        self.csv_reader = csv.DictReader(open(param.data_path, "r"))
        for row in self.csv_reader:
            date = self.row_to_datetime(row)
            date_index = (date - self.date_min).days

            x, y = self.row_to_coordinates(row)
            
            if x is None or y is None:
                continue        
            x_index = int((x - self.x_min) / self.grid_width)
            y_index = int((y - self.y_min) / self.grid_height)
            self.frames[date_index][x_index][self.width -1 -y_index] = self.frames[date_index][x_index][self.width -1 -y_index] + 1
 
    def row_to_datetime(self, row):
        return datetime.strptime(row["CRASH_DATE"].split("T")[0], "%Y-%m-%d")

    def row_to_coordinates(self, row):
        if str(row["wkt_geom"]).strip() != "NULL":
            x = abs(float(row["wkt_geom"].split(" ")[1][1:]))
            y = abs(float(row["wkt_geom"].split(" ")[2][0: -1]))
            return x, y
        else:
            return None, None

    def get_frame(self, index):
        if index < 0 or index >= len(self.frames):
            return [[0 for i in range(self.width)] for j in range(self.height)]
        else:
            return self.frames[index]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        """
        Return the samples and lables at index.

        Return:
            closeness: The closeness sequences.
            period: The period sequences.
            trend: The trend sequences.

            predict: The prediced frame.
        """
        # by days
        closeness = [self.get_frame(index - i) for i in range(param.closeness_sequence_length)]
        closeness = np.array(closeness)
        closeness = np.transpose(closeness, [1, 2, 0])
        # by weeks
        period = [self.get_frame(index - i * 7) for i in range(param.period_sequence_length)]
        period = np.array(period)
        period = np.transpose(period, [1, 2, 0])

        # by month
        trend = [self.get_frame(index - i * 30) for i in range(param.trend_sequence_length)]
        trend = np.array(trend)
        trend = np.transpose(trend, [1, 2, 0])

        # prediction
        predict = np.array(self.get_frame(index + 1))
        predict = np.expand_dims(predict, axis=2)
        return closeness, period, trend, predict
