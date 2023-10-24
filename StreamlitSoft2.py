import folium
import pandas as pd # library for data analsysis
import json # library to handle JSON files
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import requests # library to handle requests
import numpy as np
import streamlit as st
from streamlit_folium import folium_static
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pandas.api.types import is_numeric_dtype
#------------------------------------------------]
st.title(' ارزیابی منطقه')

json1 =json.load(open('C:/e/GEO_MAR/Article/filegeo/chain/final/Cell/Soft11.geojson'))
map_sby =folium.Map(location=[35.74351, 51.55231], tiles='CartoDB positron',name="Light Map",
           zoom_start=13.5,
           attr='My Data Attribution')
#json1=f"E:/save/py_test/thr.geojson"
#def center():
    #address = 'Tehran'
    #geolocator = Nominatim(user_agent="my_app")
    #location = geolocator.geocode(address)
    #latitude = location.latitude
    #longitude = location.longitude
    #return latitude, longitude
data_all = pd.read_csv('C:/e/GEO_MAR/Article/filegeo/chain/final/Cell/Soft11.csv')
choice = ['BuyingPower', 'NewChain','PopDenc','ChainDenc', 'HAFT', 'CANBO_001', 'CANBO_002',
       'CANBO_003', 'CANBO_004','CANBO_005', 'O.K_005',
'O.K_006', 'O.K_007', 'O.K_001', 'O.K_002',
       'O.K_003', 'O.K_004', 'V1_001', 'V1_002', 'V1_003', 'V1_004',  'REHAH_001','REHAH_002', 'WINMAK_001', 'WINMAK_002',
        'DAILYMAK_001', 'DAILYMAK_002' ]
choice_selected = st.selectbox('Select choice', choice)
folium.Choropleth(
    geo_data = json1,
    name="Choropleth",
    data=data_all,
    columns=['id',choice_selected],
    key_on='feature.properties.id',
    fill_color='YlOrRd', 
    fill_opacity=1, 
    line_opacity=0.00001,
    highlight=True,
    legend_name=choice_selected+"(%)",
    ).add_to(map_sby)
folium.features.GeoJson('C:/e/GEO_MAR/Article/filegeo/chain/final/Cell/Soft11.geojson',name="LSOA Code",popup=folium.features.GeoJsonPopup(fields=['BuyingPower','PopDenc'])).add_to(map_sby)
folium_static(map_sby,width=1000,height=450)

###################
st.title('ارزیابی  موقعیت فروشگاه ها')
df = pd.read_csv('C:/e/GEO_MAR/Article/filegeo/chain/final/chain2/Feature2/Join400/B400Join.csv')


if st.checkbox('Show dataframe'): 
     st.write(df)
     ##id2
Employee = st.multiselect('Select chain', df['ID'].unique())
col1 = st.selectbox('Select x column?', df.columns.unique())
col2 = st.selectbox('Select y column?', df.columns)
new_df = df[(df['ID'].isin(Employee))]
st.write(new_df)

fig = px.bar(new_df, x =col1,y=col2)

st.plotly_chart(fig)

import streamlit as st
import numpy as np
from PIL import Image

st.title(' نقشه دیجتال')

img_file_buffer = st.file_uploader("دریافت نقشه دیجتال")
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_array = np.array(image) # if you want to pass it to OpenCV
    st.image(image, caption="The caption", use_column_width=True)




#st.sidebar.header(" مدل تحلیل موقعیت فروشگاه‏های زنجیره ای")
st.sidebar.info("مدل تحلیل موقعیت فروشگاه‏های زنجیره ای")











