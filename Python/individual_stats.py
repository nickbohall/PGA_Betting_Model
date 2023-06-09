import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


class IndividualStats:
    def __init__(self):
        options = Options()
        options.headless = False  # This is to not actually open the webpage
        self.driver = webdriver.Chrome(options=options, service=Service(ChromeDriverManager().install()))

    # This is a helper function to check the url and get the column headers to create the df
    def get_column_headers(self, url):
        try:
            url = url
            self.driver.get(url)
            time.sleep(2)

            data_table = self.driver.find_elements(By.CSS_SELECTOR, "thead.css-0 th")
            column_headers = [item.text for item in data_table]
            column_headers[1] = "TREND"
            if column_headers[3] != "AVERAGE":
                column_headers[3] = "AVERAGE"
            return column_headers
        except:
            print("Failed to get column headers")

    # This function takes a URL. It has to be in the same format as most of the stats tables.
    # It will return a DF with all the table information summarized. Call multiple times to get different stats
    def get_scoring_table(self, url):
        try:
            # getting the Data In and waiting for it to open
            column_headers = self.get_column_headers(url)

            self.driver.get(url)
            time.sleep(5)

            # finding the Data In table and iterating through
            data_table = self.driver.find_elements(By.CSS_SELECTOR, "tbody.css-0 tr")

            data_table_list = []
            for row in data_table:
                # Initialize individual Dict
                player_stats_dict = {}

                player_data = row.find_elements(By.CSS_SELECTOR, "td")
                player_data_clean = [item.text.replace("Expand Row", "").replace(",","").strip() for item in player_data]

                try:
                    for i in range(len(column_headers) - 1):
                        item = player_data_clean[i]
                        try:
                            item = float(item)
                        except ValueError:
                            item = item
                        player_stats_dict[column_headers[i]] = item
                except IndexError:
                    continue

                # Add dict to list to be put in df
                data_table_list.append(player_stats_dict)

            df = pd.DataFrame(data_table_list)
            return df
        except:
            print(f"The program failed to find the table at {url}")

    # return reduced version of the tables for joins
    def get_scoring_table_condensed(self, metric, url):
        df = self.get_scoring_table(url)
        condensed_df = df[["PLAYER", "AVERAGE"]]
        condensed_df = condensed_df.rename(columns={"AVERAGE": f"{metric} Average", "PLAYER": "player"})
        print(f"{metric} df successfully gathered")
        return condensed_df










