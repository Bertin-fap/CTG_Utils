__all_ = ["built_lat_long",
          "plot_club_38",
          "plot_ctg",
          "plot_vignoble",
          "stat_sorties_club",]
          
from CTG_Utils.CTG_config import GLOBAL   

def plot_club_38():

    # Standard library import
    from pathlib import Path

    # 3rd party imports
    import folium
    import numpy as np
    import pandas as pd


    df = pd.read_excel(GLOBAL['ROOT'] / Path('club_38.xlsx'))
    path_villes_de_france = Path(__file__).parent / Path('CTG_RefFiles/villes_france_premium.csv')
    df_villes = pd.read_csv(path_villes_de_france,header=None,usecols=[2,19,20])
    dic_long = dict(zip(df_villes[2] , df_villes[19]))
    dic_lat = dict(zip(df_villes[2] , df_villes[20]))

    #df =pd.read_excel(root / Path(effectif))

    df['Ville'] = df['Ville'].str.replace(' ','-')
    df['Ville'] = df['Ville'].str.replace('ST-','SAINT-')
    df['Ville'] = df['Ville'].str.replace('\-D\-+',"-D'",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LA-',"LA ",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LE-',"LE ",regex=True)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE-DU-TOUVET',"SAINT-HILAIRE",regex=False)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE',"SAINT-HILAIRE-38",regex=False)
    df['Ville'] = df['Ville'].str.replace('LAVAL',"LAVAL-38",regex=False)
    df['Ville'] = df['Ville'].str.replace('LES-ABRETS',"LES ABRETS",regex=False)
    df['Ville'] = df['Ville'].str.lower()

    df['long'] = df['Ville'].map(dic_long)
    df['lat'] = df['Ville'].map(dic_lat)



    kol = folium.Map(location=[45.2,5.7], tiles='openstreetmap', zoom_start=12)

    for latitude,longitude,size, ville, num_ffct, club in zip(df['lat'],
                                                        df['long'],
                                                        df['number'],
                                                        df['Ville'],
                                                        df['N° FFCT'],
                                                        df['Nom Club'] ):

        long_ville, lat_ville =df.query("Ville==@ville")[['long','lat']].values[0]#.flatten()
        color='blue'

        folium.Circle(
                        location=[latitude, longitude],
                        radius=size*10,
                        popup=f'{ville} ({size}), club:{club} ',
                        color=color,
                        fill=True,
    ).add_to(kol)
    return kol
    
def built_lat_long(df):
    
    # Standard library imports
    from collections import Counter
    from pathlib import Path
    
    # 3rd party imports
    import folium
    import numpy as np
    import pandas as pd
    
    path_villes_de_france = Path(__file__).parent / Path('CTG_RefFiles/villes_france_premium.csv')
    df_villes = pd.read_csv(path_villes_de_france,header=None,usecols=[3,19,20])
    dic_long = dict(zip(df_villes[3] , df_villes[19]))
    dic_lat = dict(zip(df_villes[3] , df_villes[20]))

    df['Ville'] = df['Ville'].str.replace(' ','-')
    df['Ville'] = df['Ville'].str.replace('ST-','SAINT-')
    df['Ville'] = df['Ville'].str.replace('\-D\-+',"-D'",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LA-',"LA ",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LE-',"LE ",regex=True)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE-DU-TOUVET',"SAINT-HILAIRE",regex=False)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE',"SAINT-HILAIRE-38",regex=False)
    df['Ville'] = df['Ville'].str.replace('LAVAL',"LAVAL-38",regex=False)

    df['long'] = df['Ville'].map(dic_long)
    df['lat'] = df['Ville'].map(dic_lat)
    list_villes = df['Ville'].tolist()
    Counter(list_villes)
    dg = df.groupby(['Ville']).count()['N° Licencié']

    dh = pd.DataFrame.from_dict({'Ville':dg.index,
                                'long':dg.index.map(dic_long),
                                'lat':dg.index.map(dic_lat),
                                'number':dg.tolist()})
    return df,dh
    
def plot_ctg(df):
    
    # 3rd party imports
    import folium

    _,dh = built_lat_long(df)

    group_adjacent = lambda a, k: list(zip(*([iter(a)] * k))) 

    dict_cyclo = {}
    for ville,y in df.groupby(['Ville'])['Nom']:
        chunk = []
        for i in range(0,len(y),3):
            chunk.append(','.join(y[i:i+3] ))

        dict_cyclo[ville] = '\n'.join(chunk)

    kol = folium.Map(location=[45.2,5.7], tiles='openstreetmap', zoom_start=12)

    long_genoble, lat_grenoble = dh.query("Ville=='GRENOBLE'")[['long','lat']].values.flatten()

    folium.Circle(
                location=[lat_grenoble, long_genoble],
                radius=51000,
                popup='50 km ',
                color="black",
                fill=False,
            ).add_to(kol)        
    for latitude,longitude,size, ville in zip(dh['lat'],dh['long'],dh['number'],dh['Ville']):

        long_ville, lat_ville =dh.query("Ville==@ville")[['long','lat']].values.flatten()
        dist_grenoble_ville = _distance(lat_grenoble, long_genoble,lat_ville, long_ville )
        color='red' if dist_grenoble_ville>50  else 'blue'
        if ville == "grenoble":
            folium.Circle(
                location=[latitude, longitude],
                radius=size*50,
                popup=f'{ville} ({size}): {dict_cyclo[ville]} ',
                color="yellow",
                fill=True,
            ).add_to(kol)
        else:
                folium.Circle(
                location=[latitude, longitude],
                radius=size*100,
                popup=f'{ville} ({size}): {dict_cyclo[ville]}',
                color=color,
                fill=True,
            ).add_to(kol)
    return kol
    
def stat_sorties_club(path_sorties_club):

    import os
    
    import matplotlib.pyplot as plt 
    
    from CTG_Utils.CTG_effectif import read_effectif
    from CTG_Utils.CTG_effectif import count_participation
    from CTG_Utils.CTG_effectif import parse_date
    
    no_match,df_total,_ = count_participation(path_sorties_club)
    if no_match:
        print(no_match)

    df_effectif = read_effectif()
    dic_age = dict(zip(df_effectif['N° Licencié'], df_effectif['Age']))
    dic_distance = dict(zip(df_effectif['N° Licencié'], df_effectif['distance']))
    dic_sexe = dict(zip(df_effectif['N° Licencié'], df_effectif['Sexe']))
    dic_sexe[None] = 'irrelevant'
    dic_vae =dict(zip(df_effectif['N° Licencié'],df_effectif['Pratique VAE']))

    df_total['Age'] = df_total['N° Licencié'].map(dic_age)
    df_total['distance'] = df_total['N° Licencié'].map(dic_distance)
    df_total['sexe'] = df_total['N° Licencié'].map(dic_sexe)

    df_total = df_total[df_total['sejour']!='aucun' ]
    df_total['sejour'] = df_total['sejour'].apply(lambda s:parse_date(s,GLOBAL['YEAR']).strftime('%m-%d'))
    df_total['VAE'] = df_total['N° Licencié'].map(dic_vae)
    df_total['VAE'].fillna('Non',inplace=True)
    
    dg = df_total.groupby(['sexe','VAE'])['sejour'].value_counts().unstack().T
    try:
        dg['irrelevant'] = dg['irrelevant'] - 1
    except KeyError as error:
        pass
        
    fig, ax = plt.subplots()
    dg[['F','M']].plot(kind='bar', ax=ax, width=0.5, stacked=True)
    plt.ylabel('Nombre de licenciers')
    plt.xlabel('')
    plt.tick_params(axis='x', rotation=90)
    plt.ylabel('Nombre de licenciers')
    plt.xlabel('')
    plt.tick_params(axis='x', rotation=90)
    plt.title(os.path.basename(path_sorties_club).split('.')[0])
    
def plot_vignoble():

    # Standard library import
    from pathlib import Path

    # 3rd party imports
    import folium
    import numpy as np
    import pandas as pd


    df_club_38 = pd.read_excel(GLOBAL['ROOT'] / Path('club_38.xlsx'))
    path_villes_de_france = Path(__file__).parent / Path('CTG_RefFiles/villes_france_premium.csv')
    df_villes = pd.read_csv(path_villes_de_france,header=None,usecols=[2,19,20])
    dic_long = dict(zip(df_villes[2] , df_villes[19]))
    dic_lat = dict(zip(df_villes[2] , df_villes[20]))
    df_vignobles = pd.read_excel(r'C:\Users\franc\CTG\RANDONNEES\vignobles\listing_participants.xlsx')
    df = pd.merge(df_vignobles, df_club_38, on='N° FFCT', how='inner')
    df['femmes'] = df['femmes'].fillna(0)
    df['total'] =df['hommes']+df['femmes']

    #df =pd.read_excel(root / Path(effectif))

    df['Ville'] = df['Ville'].str.replace(' ','-')
    df['Ville'] = df['Ville'].str.replace('ST-','SAINT-')
    df['Ville'] = df['Ville'].str.replace('\-D\-+',"-D'",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LA-',"LA ",regex=True)
    df['Ville'] = df['Ville'].str.replace('^LE-',"LE ",regex=True)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE-DU-TOUVET',"SAINT-HILAIRE",regex=False)
    df['Ville'] = df['Ville'].str.replace('SAINT-HILAIRE',"SAINT-HILAIRE-38",regex=False)
    df['Ville'] = df['Ville'].str.replace('LAVAL',"LAVAL-38",regex=False)
    df['Ville'] = df['Ville'].str.replace('LES-ABRETS',"LES ABRETS",regex=False)
    df['Ville'] = df['Ville'].str.lower()

    df['long'] = df['Ville'].map(dic_long)
    df['lat'] = df['Ville'].map(dic_lat)


    kol = folium.Map(location=[45.2,5.7], tiles='openstreetmap', zoom_start=12)

    for latitude,longitude,size, ville, num_ffct, club in zip(df['lat'],
                                                        df['long'],
                                                        df['total'],
                                                        df['Ville'],
                                                        df['N° FFCT'],
                                                        df['total'] ):

        long_ville, lat_ville =df.query("Ville==@ville")[['long','lat']].values[0]#.flatten()
        color='blue'

        folium.Circle(
                        location=[latitude, longitude],
                        radius=size*70,
                        popup=f'{ville} ({size}), club:{club} ',
                        color=color,
                        fill=True,
    ).add_to(kol)
    return kol

def _distance(ϕ1, λ1,ϕ2, λ2):
    from math import asin, cos, radians, sin, sqrt
    ϕ1, λ1 = radians(ϕ1), radians(λ1)
    ϕ2, λ2 = radians(ϕ2), radians(λ2)
    rad = 6371
    dist = 2 * rad * asin(
                            sqrt(
                                sin((ϕ2 - ϕ1) / 2) ** 2
                                + cos(ϕ1) * cos(ϕ2) * sin((λ2 - λ1) / 2) ** 2
                            ))
    return dist