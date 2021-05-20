import streamlit as st
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from colorizers import *
from PIL import Image
import requests
import streamlit.components.v1 as components

def process_image(img):
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(torch.cuda.is_available()):
    	tens_l_rs = tens_l_rs.cuda()
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())
    return out_img_siggraph17

colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(torch.cuda.is_available()):
	colorizer_siggraph17.cuda()

def fix_channels(img):
    if len(img.shape)==2:
        return np.dstack((img,img,img))
    else:
        return img


##################################### AI part end ############################################

st.beta_set_page_config(page_title='AI colorizer')

# st.title('Image colorizer')

def small_title(x):
    text = f'''<p style="background: -webkit-linear-gradient(#00C9FF, #92FE9D);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        font-family: sans-serif;
                        font-weight: bold;
                        font-size:18px">
                        {x}
                        </p>'''
    return text

def html_links(text, link):
    return f'''<a href="{link}" target="_blank">{text}</a>'''

style = '''font_size: 14px;
           color: #aaa;'''

st.sidebar.title("About")
img_width = '60px'

text = f'''{small_title('The webapp')}
<p style="{style}">This webapp uses AI to color black and white images.
Users can submit a black and white image either as a file or paste the url link (make sure that the url ends with an image file extension).</p>
{small_title('References')}
<p style="{style}">The CNN architecture used in this project is inspired by the work of Richard Zhang, Phillip Isola, Alexei A. Efros in their paper {html_links('Colorful Image Colorization', 'https://arxiv.org/abs/1603.08511')}.
Checkout their amazing work on {html_links('github', 'https://github.com/richzhang/colorization')}.</p>
{small_title('The developer')}
<p style="{style}">I am a data lover who loves to create impactful tools that could help people make this world a better place.</p>
<div>
<a href="https://github.com/SarthakV7/" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/github.svg" width={img_width}"></a>
<a href="https://www.kaggle.com/sarthakvajpayee" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/kaggle.svg" width={img_width}"></a>
<a href="https://www.linkedin.com/in/sarthak-vajpayee/" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/linkedin.svg" width={img_width}"></a>
<a href="https://medium.com/@itssarthakvajpayee/" target="_blank"><img src="https://raw.githubusercontent.com/SarthakV7/covid-19-dashboard/master/assets/images/medium.png" width={img_width}"></a>
</div>
'''
st.sidebar.markdown(text, unsafe_allow_html=True)

st.markdown('''<p style="font-size: 72px;
                background: -webkit-linear-gradient(#00C9FF, #92FE9D);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-family: sans-serif;
                font-weight: bold;
                font-size:40px">
                AI image colorizer.
                </p>''', unsafe_allow_html=True)

st.subheader('Please upload a image or submit image url link')

with st.form(key='uploader'):
    uploaded_file = st.file_uploader("Choose a file...")
    url = st.text_input('Paste the image url link')
    submit_button_upl = st.form_submit_button(label='Submit image')

if (uploaded_file is None and url is None and submit_button_upl):
    st.subheader('Something\'s not right, please refresh the page and retry!')

elif (uploaded_file and url and submit_button_upl):
    st.subheader('Please select any one method to submit the image. Refresh the page and retry!')

elif url and submit_button_upl:
    img = Image.open(requests.get(url, stream=True).raw)
    img = np.array(img)
    img = fix_channels(img)
    with st.spinner(f'Colorizing image please wait...'):
        out_img = process_image(img)
        st.image(out_img, caption='colorized image')
        st.image(img, caption='input image')
        # st.subheader(f'input image size:{img.shape}\noutput image size:{out_img.shape}')

elif uploaded_file and submit_button_upl:
    img = Image.open(uploaded_file)
    img = np.array(img)
    img = fix_channels(img)
    with st.spinner(f'Colorizing image please wait...'):
        out_img = process_image(img)
        st.image(out_img, caption='colorized image')
        st.image(img, caption='input image')
        # st.subheader(f'input image size:{img.shape}\noutput image size:{out_img.shape}')
