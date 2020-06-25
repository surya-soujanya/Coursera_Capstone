#!/usr/bin/env python
# coding: utf-8

# # Capstone Project
# ## Applied Data Science Capstone by IBM/Coursera
# ### Sydney - Food and Travel
# 
# *By Surya Soujanya Kodavalla*

import pandas as pd
import numpy as np
import streamlit as st
import requests
import csv
from PIL import Image
import json
from pandas.io.json import json_normalize

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
import urllib
import folium
from geopy.distance import great_circle
import altair as alt
print('Libraries imported.')

st.title('Sydney — Food and Travel')

section = st.sidebar.selectbox(
    'Which part would you like to see?',
    ('Introduction', 'Data', 'Clustering', 'Results', 'Future Direction'))
st.sidebar.markdown("[Medium](https://medium.com/@soujanya927/sydney-food-and-travel-de0fe00d4a32)")
st.sidebar.markdown("[Github](https://github.com/surya-soujanya/Coursera_Capstone)")
st.sidebar.markdown("[LinkedIn profile of author](https://www.linkedin.com/in/suryasoujanya/)")
    
sydney_suburbs = pd.read_csv('sydney_suburbs.csv')
sydney_stations = pd.read_csv('sydney_stations.csv')

#Obtain the latitude and longitude of Sydney
@st.cache
def sydney_loc():
    address = 'Sydney, Australia'
    geolocator = Nominatim(user_agent="Sydney_explorer")
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude,longitude
x,y = sydney_loc()
print('The geograpical coordinate of Sydney are {}, {}.'.format(x,y))
st.subheader(section)

top_food_places_original = pd.read_csv('top_food_places_initial.csv')
top_food_places = pd.read_csv('top_food_places.csv')

clustering = pd.DataFrame(columns=['Distance to Nearest Station in km'])
clustering['Distance to Nearest Station in km'] = top_food_places['Distance to Nearest Station in km']
kmeans = KMeans(n_clusters=5, random_state=0).fit(clustering)
clusters = kmeans.labels_
top_food_places.insert(loc=9,column="Cluster Labels",value=clusters)

cluster_0 = pd.DataFrame()
cluster_0 = top_food_places.loc[(top_food_places['Cluster Labels'] == 0)]
describe_0 = cluster_0['Distance to Nearest Station in km'].describe()

cluster_1 = pd.DataFrame()
cluster_1 = top_food_places.loc[(top_food_places['Cluster Labels'] == 1)]
describe_1 = cluster_1['Distance to Nearest Station in km'].describe()

cluster_2 = pd.DataFrame()
cluster_2 = top_food_places.loc[(top_food_places['Cluster Labels'] == 2)]
describe_2 = cluster_2['Distance to Nearest Station in km'].describe()

cluster_3 = pd.DataFrame()
cluster_3 = top_food_places.loc[(top_food_places['Cluster Labels'] == 3)]
describe_3 = cluster_3['Distance to Nearest Station in km'].describe()

cluster_4 = pd.DataFrame()
cluster_4 = top_food_places.loc[(top_food_places['Cluster Labels'] == 4)]
describe_4 = cluster_4['Distance to Nearest Station in km'].describe()

mins = [describe_0.iloc[3], describe_1.iloc[3], describe_2.iloc[3], describe_3.iloc[3], describe_4.iloc[3]]
maxs = [describe_0.iloc[7], describe_1.iloc[7], describe_2.iloc[7], describe_3.iloc[7], describe_4.iloc[7]]
means = [describe_0.iloc[1], describe_1.iloc[1], describe_2.iloc[1], describe_3.iloc[1], describe_4.iloc[1]]
cluster_names = ['Cluster 0', 'Cluster 1','Cluster 2','Cluster 3','Cluster 4']
d = {"Clusters": cluster_names, "Minimum distance in each cluster": mins}
min_data = pd.DataFrame(d,columns = ["Clusters","Minimum distance in each cluster"])
d = {"Clusters": cluster_names, "Maximum distance in each cluster": maxs}
max_data = pd.DataFrame(d, columns = ["Clusters","Maximum distance in each cluster"])
d = {"Clusters": cluster_names, "Average distance in each cluster": means}
mean_data = pd.DataFrame(d, columns = ["Clusters","Average distance in each cluster"])
mean_chart_st = pd.DataFrame(means, index = cluster_names, columns = ["Average distance in each cluster"])
min_cluster = mins.index(min(mins))

results_suburbs = pd.DataFrame(columns = ["Suburb","Cluster"])
results_stations = pd.DataFrame(columns = ["Station","Cluster"])

results_suburbs["Suburb"] = top_food_places["Suburb"]
results_stations["Station"] = top_food_places["Nearest Station"]
results_suburbs["Cluster"] = top_food_places["Cluster Labels"]
results_stations["Cluster"] = top_food_places["Cluster Labels"]

suburbs_uni = pd.DataFrame(columns = ["Suburb"])
stations_uni = pd.DataFrame(columns = ["Station"])

suburbs_uni["Suburb"] = results_suburbs["Suburb"].unique()
stations_uni["Station"] = results_stations["Station"].unique()

for i in range(0,5):
    df = results_suburbs.loc[(results_suburbs["Cluster"] == i)]
    df = df["Suburb"].value_counts()
    df = df.reset_index()
    df.rename(columns={"index": "Suburb", "Suburb": "Cluster "+str(i)},inplace = True)
    df.sort_values(by=["Suburb"],inplace = True)
    df = df.reset_index()
    df.drop(['index'],inplace=True,axis=1)
    suburbs_uni = pd.merge(suburbs_uni, df, how="outer", on=["Suburb"])
suburbs_uni.replace(np.nan, 0, inplace = True)
cols = suburbs_uni.columns
min_cluster_col_subs = cols[min_cluster+1]
#suburbs_uni.sort_values(by=[min_cluster_col_subs],inplace = True,ascending = False)
#suburbs_uni.reset_index(inplace = True)
#suburbs_uni.drop(['index'],inplace=True,axis=1)


for i in range(0,5):
    df = results_stations.loc[(results_stations["Cluster"] == i)]
    df = df["Station"].value_counts()
    df = df.reset_index()
    df.rename(columns={"index": "Station", "Station": "Cluster "+str(i)},inplace = True)
    df.sort_values(by=["Station"],inplace = True)
    df = df.reset_index()
    df.drop(['index'],inplace=True,axis=1)
    stations_uni = pd.merge(stations_uni, df, how="outer", on=["Station"])
stations_uni.replace(np.nan, 0, inplace = True)
cols = stations_uni.columns
min_cluster_col_stats = cols[min_cluster+1]
stations_uni.sort_values(by=[min_cluster_col_stats],inplace = True,ascending = False)
stations_uni.reset_index(inplace = True)
stations_uni.drop(['index'],inplace=True,axis=1)






if section == 'Introduction':
    image = Image.open('img.jpeg')
    st.markdown('The purpose of this project is to identify the trending food ventures in each suburb of Sydney, Australia with accessible public transportation. Sydney has around 600 plus suburbs in the city with an excellent metro system and an amazing variety of cuisines available to gorge on. By calculating the distance from the nearest train stations and clustering restaurants based on this distance, anyone new to the beautiful city of Sydney can enjoy the food there without much of a hassle about transport.')
    st.image(image, caption='Opera House and the Harbour Bridge',use_column_width=True)
    st.markdown('The above problem has been addressed by finding the trending food places in each suburb of Sydney which are closest to train stations. Specifically, this report will be targeted to travelers interested in visiting the top food places which are are convenient to travel to and from in Sydney. We are particularly focusing on the train stations in Sydney as their metro network is very reliable with services running from 4am to around midnight on most train lines.')
    st.markdown('To do the analysis as mentioned above, initially a list of all the suburbs, all the train stations and the trending restaurants in Sydney are obtained. From this data, the nearest train station to each restaurant is found based on their locations and the distances calculated. After clustering the restaurants based on the distances previously calculated, we can come to a conclusion about the most accessible restaurants, etc.')
    st.markdown('Data science methodologies and the K-means clustering algorithm have been used to get useful data and generate different clusters of food places all over Sydney based on their distances from the nearest train stations.')
    
if (section == 'Data'):
    st.markdown("Based on definition of the problem, data needed for analysis is:")
    st.markdown("* list of suburbs in Sydney and their locations")
    st.markdown("* list of train stations in Sydney and their locations")
    st.markdown("* top food venues in different suburbs")
    st.markdown("* distance between food venues and the nearest train station")
    st.markdown("#### List of Suburbs in Sydney and their locations -")
    st.markdown("* Data for list of suburbs in Sydney was scraped from the [Wikipedia page](https://en.wikipedia.org/wiki/List_of_Sydney_suburbs) and it was cleaned by removing extra parts of the code from in between the data and other extra characters through multiple iterations. The scraped and cleaned list of suburbs was then made into a dataframe and stored as a csv file.")
    st.markdown("* While adding the latitudes and longitudes of each of the suburbs to the dataset, using the geopy package, the location 'Sydney, Australia' was added to the address to get the accurate location as there can possibly be many places with the same name in different cities.")
    st.markdown("* The link to the data set is available [here](https://www.kaggle.com/ssk27997/suburbs-in-sydney-australia).")
    st.markdown("* The link to the code used for scraping is available [here](https://github.com/surya-soujanya/Coursera_Capstone/blob/master/Suburbs_Scraping.ipynb).")
    show_data_suburbs = st.checkbox('Show suburbs data')
    if show_data_suburbs:
        st.write(sydney_suburbs)
    show_map_suburbs = st.checkbox('Show suburbs map')
    if show_map_suburbs:
        map_suburbs = sydney_suburbs
        map_suburbs.drop(['Suburb'],inplace=True,axis=1)
        map_suburbs.rename(columns = {"Latitude":"lat","Longitude":"lon"},inplace = True)
        #st.write(map_suburbs)
        st.map(map_suburbs)

    st.markdown("#### List of Train Stations in Sydney and their locations -")
    st.markdown("* Similar to the dataset for the suburbs, the dataset for the railway stations was scraped using beautifulsoup from the  [Wikipedia page](https://en.wikipedia.org/wiki/List_of_Sydney_Trains_railway_stations). It was cleaned by removing extra parts of the code from in between the data and other extra characters through multiple iterations. The words 'Railway Station' were added to each of the names of the stations to make locating them easier. The scraped and cleaned list of railway stations was then made into a dataframe and stored as a csv file.")
    st.markdown("* While adding the latitudes and longitudes of each of the stations to the dataset, using the geopy package, the location 'Sydney, Australia' was added to the address to get the accurate location as there can possibly be many places with the same name in different cities.")
    st.markdown("* The link to the data set is available [here](https://www.kaggle.com/ssk27997/train-stations-in-sydney-australia).")
    st.markdown("* The link to the code used for scraping is available [here](https://github.com/surya-soujanya/Coursera_Capstone/blob/master/Stations_Scraping.ipynb).")
    show_data_stations = st.checkbox('Show stations data')
    if show_data_stations:
        st.write(sydney_stations)
    show_map_stations = st.checkbox('Show stations map')
    if show_map_stations:
        map_stations = sydney_stations
        map_stations.drop(['Station'],inplace=True,axis=1)
        map_stations.rename(columns = {"Latitude":"lat","Longitude":"lon"},inplace = True)
        #st.write(map_suburbs)
        st.map(map_stations)
    st.markdown("#### Top food places and their locations -")
    st.markdown("The food places in each suburb are obtained by using the Foursquare API on the suburbs dataset. The category ID for food places is used to return only the food places in the specified areas. The required details of each venue such as latitude and longitude are obtained from the json file generated.")
    show_food_venues = st.checkbox('Show top food venues data')
    if show_food_venues:
        st.write(top_food_places_original)

if (section == 'Clustering'):
    st.markdown("In this project, the plan is to cluster the food places based on their distances from the nearest railway stations.")
    st.markdown("In the first step, we collected the required data as shown above by data scraping, cleaning and using the geopy package and the Foursquare API.")
    st.markdown("The second step is the calculation and exploration of the nearest railway station to each food place and the distance between them. This data is then combined with the dataset of top food places that was previously obtained. To find the closest railway station to each restaurant, the distance between each restaurant returned by the call to the Foursquare API and train station is calculated using the geopy package. Then the station with the least distance is concluded to be the nearest station to each food place.")
    st.markdown("The third and final step will be focusing on clustering these food places based on their distances from the nearest railway stations and exploring the clusters made. A visualization of the clusters obtained will be made by plotting a few points of each cluster on a map along with the railway stations. The clusters obtained can be used to explore the restaurants and suburbs as per the requirement of the traveler.")
    st.write(top_food_places)
    
if (section == 'Results'):

    st.markdown("### Cluster Details")
    #st.bar_chart(min_chart)
    #st.bar_chart(max_chart)
    #mean_chart = alt.Chart(mean_data).mark_bar().encode(x = "Clusters", y = "Average distance in each cluster")
    #st.altair_chart(mean_chart, use_container_width=True)
    st.bar_chart(mean_chart_st)
    st.write('We can see that cluster ',mins.index(min(mins)),'has all the food places within a walkable distance to a train station while cluster ', mins.index(max(mins)),' has the places farthest from a nearby train station')
    st.write('Places in cluster ',mins.index(min(mins)),'are at most at a distance of  ',round(min(maxs), 2),' km from a train station.')
    st.write('Each cluster covers a certain range of distances.The order of clusters based on ranges of the distances they cover in increasing order are cluster',mins.index(sorted(mins)[0]),', cluster ',mins.index(sorted(mins)[1]),', cluster ',mins.index(sorted(mins)[2]),', cluster ',mins.index(sorted(mins)[3]),'and cluster ',mins.index(sorted(mins)[4]))
    
    st.markdown("### Suburb Details")

    suburb_query = st.text_input('Type in the suburb you want details of','Granville')
    return_number = suburbs_uni[min_cluster_col_subs].loc[suburbs_uni["Suburb"] == suburb_query]
    return_number = return_number.to_numpy()
    st.write('There are ',int(return_number[0]) ,'food places in ', suburb_query,' which are in walkable distance of a train station.')
   
    add_suburbs_slider = st.slider('Select a range of values',0, len(suburbs_uni["Suburb"]), (0, 39))
    st.markdown("*The data shows the number of food places in each cluster in every suburb (sorted by suburb name).*")
    begin,end = add_suburbs_slider
    suburbs_uni_show = pd.DataFrame()
    suburbs_uni_show = suburbs_uni.iloc[begin:end]
    suburbs_uni_show
    uni = suburbs_uni_show["Suburb"]
    subs = []
    for each in uni:
        for i in range(0,5):
            subs.append(each)
    clusts = []
    type=["0","1","2","3","4"]
    for j in range(len(uni)):
        for i in type:
            clusts.append(i)
    subs_clusts_0 = suburbs_uni_show["Cluster 0"]
    subs_clusts_1 = suburbs_uni_show["Cluster 1"]
    subs_clusts_2 = suburbs_uni_show["Cluster 2"]
    subs_clusts_3 = suburbs_uni_show["Cluster 3"]
    subs_clusts_4 = suburbs_uni_show["Cluster 4"]
    clusts_all = []
    for j in range(begin,end):
            clusts_all.append(subs_clusts_0[j])
            clusts_all.append(subs_clusts_1[j])
            clusts_all.append(subs_clusts_2[j])
            clusts_all.append(subs_clusts_3[j])
            clusts_all.append(subs_clusts_4[j])
    suburbs_chart_data = pd.DataFrame(columns = ["Suburb","Cluster Number","Number of Food Places"])
    suburbs_chart_data["Suburb"]=subs
    suburbs_chart_data["Cluster Number"]=clusts
    suburbs_chart_data["Number of Food Places"]=clusts_all
    c=alt.Chart(suburbs_chart_data).mark_bar().encode(x='Suburb', y='Number of Food Places',color = 'Cluster Number')
    st.write(c)
    
    st.markdown("### Station Details")

    station_query = st.text_input('Type in the station you want details of','Wentworthville Railway Station')
    return_number = stations_uni[min_cluster_col_stats].loc[stations_uni["Station"] == station_query]
    return_number = return_number.to_numpy()
    st.write('There are ',int(return_number[0]) ,'food places near ', station_query,' within walkable distance.')

    add_stations_slider = st.slider('Select a range of values',0, len(stations_uni["Station"]), (0, 39))
    st.markdown("*The data shows the number of food places near the respective railway station in each cluster (sorted by station name).*")
    begin_1,end_1 = add_stations_slider
    stations_uni_show = pd.DataFrame()
    stations_uni_show = stations_uni.iloc[begin_1:end_1]
    stations_uni_show
    uni = stations_uni_show["Station"]
    stats = []
    for each in uni:
        for i in range(0,5):
            stats.append(each)
    clusts = []
    type=["0","1","2","3","4"]
    for j in range(len(uni)):
        for i in type:
            clusts.append(i)
            
    stats_clusts_0 = stations_uni["Cluster 0"]
    stats_clusts_1 = stations_uni["Cluster 1"]
    stats_clusts_2 = stations_uni["Cluster 2"]
    stats_clusts_3 = stations_uni["Cluster 3"]
    stats_clusts_4 = stations_uni["Cluster 4"]
    clusts_all = []
    for j in range(begin_1,end_1):
            clusts_all.append(stats_clusts_0[j])
            clusts_all.append(stats_clusts_1[j])
            clusts_all.append(stats_clusts_2[j])
            clusts_all.append(stats_clusts_3[j])
            clusts_all.append(stats_clusts_4[j])
    stations_chart_data = pd.DataFrame(columns = ["Station","Cluster Number","Number of Food Places"])
    stations_chart_data["Station"]=stats
    stations_chart_data["Cluster Number"]=clusts
    stations_chart_data["Number of Food Places"]=clusts_all

    c=alt.Chart(stations_chart_data).mark_bar().encode(x='Station', y='Number of Food Places',color = 'Cluster Number')
    st.write(c)
    

if (section == 'Future Direction'):
     st.markdown("This project can be further extended by creating analysis of food places closest to both train stations and the iconic places to visit in Sydney like the Opera House and its famous beaches.")
     st.markdown("The same analysis can be done on other cities as well.")
    #st.markdown("*_Please note that it is only for this specific run of the code and the restaurants given for this specific call of the Foursquare API that cluster 0 has the restaurants within walkable distance of the train stations with the measures given above. It is possible that if the code is run again, the call to the Foursquare API may give different results depending on the trending venues at that particular point of time which may change the details of each cluster obtained in cells 80 to 84 in the code. It is not necessary that cluster 0 always has the minimum distance restaurants. It is important to understand how the clustering algorithm works on the distances array given as the input — the algorithm divides the different distances into different clusters so that each cluster covers a particular range of distances. On studying the details of each cluster given by the describe function as shown above, it can be understood as to which range is covered by which cluster._*")
     st.markdown("# Bon Voyage and Bon Appetit !")
