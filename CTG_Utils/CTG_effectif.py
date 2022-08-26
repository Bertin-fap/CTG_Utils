__all__ = ['correction_effectif',
           'count_participation',
           'inscrit_sejour',
           'parse_date',
           'read_effectif',
           'read_effectif_corrected',]
           
from CTG_Utils.CTG_config import GLOBAL           

def read_effectif_corrected(dic_correction_licence, list_non_licencie, part_club):
    
    '''Lecture du fichier effectif et correction
    '''
    # Standard library imports
    from pathlib import Path
    
    # 3rd party imports
    import numpy as np
    import pandas as pd
    
    prenom, nom, sexe = zip(*list_non_licencie)
    dict_non_licencie = {'N° Licencié':np.array(range(len(nom)))+10, 'Prénom':prenom,'Nom':nom,'Sexe':sexe}
    
    prenom, nom, sexe = zip(*part_club)
    part_club = {'N° Licencié':list(range(len(part_club))), 'Prénom':prenom,'Nom':nom,'Sexe':sexe}

    df_effectif = pd.read_excel(GLOBAL['ROOT'] / Path(GLOBAL['EFFECTIF']))
    df_effectif = df_effectif[['N° Licencié', 'Nom','Prénom','Sexe']]
    
        
    for num_licence in dic_correction_licence.keys():
        idx = df_effectif[df_effectif["N° Licencié"]==num_licence].index
        df_effectif.loc[idx,'Prénom'] = dic_correction_licence[num_licence]['Prénom']
        df_effectif.loc[idx,'Nom'] = dic_correction_licence[num_licence]['Nom']
        
    df_part_club = pd.DataFrame.from_dict(part_club)
    df_effectif = pd.concat([df_effectif, df_part_club], ignore_index=True, axis=0)
    
    df_non_licencie = pd.DataFrame.from_dict(dict_non_licencie)
    df_effectif = pd.concat([df_effectif, df_non_licencie], ignore_index=True, axis=0)
    
    df_effectif['Prénom1'] = df_effectif['Prénom'].str[0]
    df_effectif['Prénom'] = df_effectif['Prénom'].str.replace(' ','-')
    
    return df_effectif

def inscrit_sejour(file,no_match,df_effectif):

    # Standard library import
    import functools
    import os
    import unicodedata
    
    # 3rd party import
    import pandas as pd


    nfc = functools.partial(unicodedata.normalize,'NFD')
    convert_to_ascii = lambda text : nfc(text). \
                                     encode('ascii', 'ignore'). \
                                     decode('utf-8').\
                                     strip()

    df = pd.read_csv(file) 
    sejour = os.path.splitext(os.path.basename(file))[0]
    
    if len(df) != 0:
        dg = df['Unnamed: 0'].str.upper()
        dg = dg.dropna()
        dg = dg.str.replace(' \t?','',regex=False)
        dg = dg.str.replace('JO ','JOSEPH ')
        dg = dg.str.replace('HERVÉ.P','PEREZ HERVE',regex=False)
        dg = dg.str.replace('MARTINE HENAULT-VAILLI','MARTINE HENAULT',regex=False)
        dg = dg.str.replace('.',' ',regex=False)
        dg = dg.apply(convert_to_ascii)
        dg = dg.drop_duplicates()
        dg = dg.str.split('\s{1,10}')

        dg = dg.apply(lambda row : row+[None] if len(row)==2 else row)

        split_dg = pd.DataFrame(dg.tolist(), columns=['name1', 'name2', 'name3'])

        dic = {}
        for idx,row in split_dg.iterrows():
            if (row.name3 is None) and ( row.name2 is not None):
                if len(row.name1)==1:
                    dr = df_effectif.query('Prénom1==@row.name1[0] and Nom==@row.name2')
                    if len(dr):
                        dic[idx] =dr.iloc[0].tolist()[:-1]+[sejour]
                    else:
                        print('no match',row.name2,row.name1)
                        no_match.append((row.name2,row.name1))
                elif len(row.name2)==1:
                    dr = df_effectif.query('Prénom1==@row.name2 and Nom==@row.name1')
                    if len(dr):
                         dic[idx] =dr.iloc[0].tolist()[:-1]+[sejour]
                    else:
                        print('no match',row.name2,row.name1)
                        no_match.append((row.name2,row.name1))
                else:
                    if len((dr:=df_effectif.query('Prénom==@row.name2 and Nom==@row.name1'))):
                         dic[idx] =dr.iloc[0].tolist()[:-1]+[sejour]
                    elif len((dr:=df_effectif.query('Prénom==@row.name1 and Nom==@row.name2'))):
                         dic[idx] =dr.iloc[0].tolist()[:-1]+[sejour]
                    else:
                        print(f'no match, prénom:{row.name2}, nom: {row.name1}',row.name2,row.name1)
                        no_match.append((row.name2,row.name1))
            else:
                print(f'WARNING: incorrect name {row.name1}, {row.name2}, {row.name3}')

        dg = pd.DataFrame.from_dict(dic).T 
        dg.columns = ['N° Licencié','Nom','Prénom','Sexe','sejour',]
    else:
        dg = pd.DataFrame([[None,None,None,None,sejour,]], columns=['N° Licencié','Nom','Prénom','Sexe','sejour',])
    
    return dg

def correction_effectif():
    
    # Standard library import
    from pathlib import Path
    
    #3rd party import
    import yaml
    
    path_cor_yaml = Path(__file__).parent / Path('CTG_RefFiles/CTG_correction.yaml')
    
    with open(path_cor_yaml, "r",encoding='utf8') as stream:
        data_list_dict = yaml.safe_load(stream)

    list_non_licencie = [(x.split(',')[0].strip(),x.split(',')[1].strip(),x.split(',')[2].strip())
                         for x in data_list_dict['list_non_licencie']]
    dic_part_club = [(x.split(',')[0].strip(),x.split(',')[1].strip(),x.split(',')[2].strip()) 
                         for x in data_list_dict['dic_part_club']]
    dic_correction_licence = data_list_dict['dic_correction_licence']
    dic_correction_licence = {list(x.keys())[0] : list(x.values())[0] for x in dic_correction_licence}
    
    return list_non_licencie, dic_correction_licence, dic_part_club

    
def count_participation(path):
    
    import os
    from pathlib import Path

    import pandas as pd

    filename = os.path.basename(path)
    filename = filename.split('.', 1)[0]
    list_non_licencie, dic_correction_licence, dic_part_club = correction_effectif()

    df_effectif = read_effectif_corrected(dic_correction_licence, list_non_licencie, dic_part_club)

    no_match = []
    df_list = [] 
    sejours = [x for x in os.listdir( path ) if x.endswith('.csv')]
    nbr_sejours = len(sejours)
    print(f"Nombre d'évènements : {nbr_sejours}")
    
    nbr_inscrits_mean = 0
    counter = 1
    for sejour in sejours:
        dg = inscrit_sejour( path / Path(sejour),no_match,df_effectif)
        dg['Type'] = filename
        nbr_inscrits = len(dg.dropna())
        if nbr_inscrits != 0:
           nbr_inscrits_mean = nbr_inscrits_mean + (nbr_inscrits - nbr_inscrits_mean)/counter
           counter += 1
        print(f"Séjour :{sejour}, Nombre d'inscrits : {nbr_inscrits}")
        df_list.append(dg)
        file_store = os.path.splitext(sejour)[0]+'.xlsx'
        dg.to_excel(path / Path(file_store))
    print(f'" moyen de participants : {nbr_inscrits_mean}')
    df_total = pd.concat(df_list,ignore_index=True)


    liste_licence = df_effectif['N° Licencié']
    liste_licence_sejour = df_total['N° Licencié']
    index = list(set(liste_licence)-set(liste_licence_sejour))

    df_non_inscrits = df_effectif.copy()
    df_non_inscrits = df_non_inscrits[df_non_inscrits['N° Licencié'].isin(index)]
    df_non_inscrits['sejour'] = 'aucun'
    df_total = pd.concat([df_total,df_non_inscrits],ignore_index=True)

    return(no_match,df_total,index)

def parse_date(s,year):
    
    import re
    from datetime import datetime 

    convert_to_date = lambda s: datetime.strptime(s,"%Y_%m_%d")

    pattern = re.compile(r"(?P<month>\b\d{1,2}[_,-])(?P<day>\d{1,2})")
    match = pattern.search(s)
    
    return convert_to_date(str(year)+'_'+match.group("month")+match.group("day"))
    
def read_effectif():
    
    from pathlib import Path
    
    import numpy as np
    import pandas as pd
    
    from CTG_Utils.CTG_plot import built_lat_long

    def distance_(row):
        from math import asin, cos, radians, sin, sqrt
        phi1, lon1 = dh.query("Ville=='GRENOBLE'")[['long','lat']].values.flatten()
        phi1, lon1 = radians(phi1), radians(lon1)
        phi2, lon2 = radians(row['long']), radians(row['lat'])
        rad = 6371
        dist = 2 * rad * asin(
                                sqrt(
                                    sin((phi2 - phi1) / 2) ** 2
                                    + cos(phi1) * cos(phi2) * sin((lon2 - lon1) / 2) ** 2
                                ))
        return np.round(dist,1)


    df =pd.read_excel(GLOBAL['ROOT'] / Path(GLOBAL['EFFECTIF']))
    df['Date de naissance'] = pd.to_datetime(df['Date de naissance'], format="%d/%m/%Y")
    df['Age']  = df['Date de naissance'].apply(lambda x : (pd.Timestamp.today()-x).days/365)
    df['Date du certificat'] = df['Date du certificat'].apply(lambda x : x if x != 'Non' else '01/01/2000')
    df['Date du certificat'] = pd.to_datetime(df['Date du certificat'], format="%d/%m/%Y")
    df['Limite certificat médical']  = df['Date du certificat'].apply(lambda x : (pd.Timestamp.today()-x).days/365)

    df,dh = built_lat_long(df)

    df['distance'] = df.apply(distance_,axis=1)
    
    return df