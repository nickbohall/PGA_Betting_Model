#Import scraping modules
import requests

#Import Data In manipulation modules
import pandas as pd
import numpy as np

#other
from datetime import date

#other files
import playerMapping
import tourneyList

year = date.today().year
start = 'https://www.pgatour.com/tournaments/' #just creating the links more efficiently, they all start with this
tourney = 'fortinet-championship' #This needs to be an array of tournaments to loop through
middle = '/past-results/jcr:content/mainParsys/pastresults.selectedYear.'
years = range(year-5, year+1)
end = '.html' #and end with this

url = f'{start}{tourney}{middle}{str(year)}{end}'
def tourneyStats(tourney):
    my_dict = {'player': {'year':[],'pos':[], 'score':[] }}
    my_df = pd.DataFrame()
    my_df['player number'] = playerMapping.player_df['player number']
    my_df['full name'] = playerMapping.player_df['full name']
    for year in years:
        url = f'{start}{tourney}{middle}{str(year)}{end}'
        try:
            r = requests.get(url).text
            t = pd.read_html(r)[0]
            t.reset_index()
            t.columns = [':'.join([i[0], i[1]]) if 'ROUNDS' in i else i[0] for i in t.columns]
            t.POS = t.POS.map(lambda x: x.split(' ')[-1])
            t['TO PAR'] = t['TO PAR'].map(lambda x: x.split(' ')[-1])
            round_columns = [i for i in t.columns if 'ROUNDS' in i]
            t[round_columns] = t[round_columns].applymap(lambda x: x.split(' ')[0])
            t = t.rename(columns={'PLAYER': 'full name', 'POS': f'{year} pos','TO PAR': f'{year} toPar'})
            df = t[['full name', f'{year} pos', f'{year} toPar']]
            numPlayers = len(df.index) #counts no. of players in the tourney
            cutFinish = str(round(numPlayers * (3/4))) #a number to punish if you didn't make the cut, set to 3/4s way down the leaderboard
            df = df.replace('CUT', cutFinish, regex=True) #replacing cut with the cutNo
            df = df.replace('T','', regex=True) #removing Ties
            df = df.replace('W/D',  np.nan , regex=True) #disregard W/D
            df = df.replace('E','0', regex=True) #Even par is 0
            df = df.replace('DQ', np.nan, regex=True) #disregard DQ
            df = df.astype({f'{year} pos': 'Int64', f'{year} toPar': 'Int64'})
            my_df = my_df.merge(df, how='left', on= 'full name')
            print(my_df.dtypes)
        except: 
            print(f'{year} did not have this tourney')
            continue
        
    my_df = my_df.dropna(thresh=3)
    posCols = [col for col in my_df.columns if 'pos' in col] #creating list of cols if they contain pos - this is because not all tourney have all years
    parCols = [col for col in my_df.columns if 'toPar' in col] #creating list of cols if they contain Par - this is because not all tourney have all years
    my_df['posAvg'] = my_df[posCols].mean(axis=1, skipna=True).round() #getting avg of position
    my_df['scoreAvg'] = my_df[parCols].mean(axis=1, skipna=True).round() #getting avg of score
    return my_df.sort_values(by="posAvg", ascending = True)

def allTourneys():    
    for tourney in tourneyList.tourneyListformatted:
        print(tourney)
        print(tourneyStats(tourney).head(5))

print(tourneyStats('fedex-st-jude-championship').head(50))