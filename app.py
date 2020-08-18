import streamlit as st

#EDA pkg
import pandas as pd
import numpy as np

# Model Load/Save
from joblib import load
import joblib
import os

#Data Viz pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ML Pkgs
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Database
import sqlite3

class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	def __init__(self,age=None ,wife_education=None ,husband_education=None ,num_of_children_ever_born=None ,result_wife_reg=None ,result_wife_working=None ,husband_occupation=None ,standard_of_living=None ,result_media_exposure=None, predicted_class=None,model_class=None):
		super(Monitor, self).__init__()
		self.age = age
		self.wife_education = wife_education
		self.husband_education = husband_education
		self.num_of_children_ever_born = num_of_children_ever_born
		self.result_wife_reg = result_wife_reg
		self.result_wife_working = result_wife_working
		self.husband_occupation = husband_occupation
		self.standard_of_living = standard_of_living
		self.result_media_exposure = result_media_exposure
		self.predicted_class = predicted_class
		self.model_class = model_class

	def __repr__(self):
		"Monitor(age = {self.age},wife_education = {self.wife_education},husband_education = {self.husband_education},num_of_children_ever_born = {self.num_of_children_ever_born},result_wife_reg = {self.result_wife_reg},result_wife_working = {self.result_wife_working},husband_occupation = {self.husband_occupation},standard_of_living = {self.standard_of_living},result_media_exposure = {self.result_media_exposure},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS cmcprediction(age NUMERIC,wife_education NUMERIC,husband_education NUMERIC,num_of_children_ever_born NUMERIC,result_wife_reg NUMERIC,result_wife_working NUMERIC,husband_occupation NUMERIC,standard_of_living NUMERIC,result_media_exposure NUMERIC,predicted_class NUMERIC,model_class TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO cmcprediction(age,wife_education,husband_education,num_of_children_ever_born,result_wife_reg,result_wife_working,husband_occupation,standard_of_living,result_media_exposure,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.wife_education,self.husband_education,self.num_of_children_ever_born,self.result_wife_reg,self.result_wife_working,self.husband_occupation,self.standard_of_living,self.result_media_exposure,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM cmcprediction')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data



		


# Functions

## Load css
def load_css(css_name):
	with open(css_name) as f:
		st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

## Load icon
def load_icon(name):
	st.markdown('<i class ="material-icons">{}</i>'.format(name), unsafe_allow_html=True)

## remote_css
def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)

## icon-css
def icon_css(icone_name):
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

## Getting value from dictionary
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

## Getting Keys from dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key

## Load Models
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Main
def main():
	""" Salary Prediction ML App with Streamlit """
	st.title("Contraceptive Method Choice Prediction")
	st.text("Predicting Contraceptive Choice with ML and Streamlit")

	# Loading Dataset
	df = pd.read_csv("cmc_dataset.csv")

	# Sidebar (TABS/ Menus)
	bars = ['EDA','Prediction','Monitor','About']
	choice = st.sidebar.selectbox("Choose Activity", bars)


	# Choice EDA
	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")
		load_css('icon.css') #function defines at the top
		load_icon('dashboard') #function defines at the top

		if st.checkbox("Show DataSet"):
			number = st.number_input("Number of Rows to View",value=5)
			st.dataframe(df.head(number))

		if st.checkbox("Columns Names"):
			st.write(df.columns)

		if st.checkbox("Shape of Dataset"):
			st.write(df.shape)
			data_dim = st.radio("Show Dimension by",("Rows","Columns"))
			if data_dim == 'Rows':
				st.text("Number of  Rows")
				st.write(df.shape[0])
			elif data_dim == 'Columns':
				st.text("Number of Columns")
				st.write(df.shape[1])

		if st.checkbox("Select Columns To Show"):
			all_columns = df.columns.tolist()
			selected_columns = st.multiselect('Select',all_columns)
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Data Types"):
			st.write(df.dtypes)

		if st.checkbox("Value Counts"):
			st.text("Value Counts By Target/Class")
			st.write(df.iloc[:,-1].value_counts())

		st.subheader("Data Visualization")
		# Show Correlation Plots
		# Matplotlib Plot
		st.markdown("#### Correlation Plot")
		if st.checkbox("Correlation Plot [Matplotlib]"):
			plt.matshow(df.corr())
			st.pyplot()
		# Seaborn Plot
		if st.checkbox("Correlation Plot with Annotation[Seaborn]"):
			st.write(sns.heatmap(df.corr(),annot=True))
			st.pyplot()

		# Counts Plots
		st.markdown("#### Value Count Plot")
		if st.checkbox("Plot of Value Counts"):
			st.text("Value Counts By Target/Class")

			all_columns_names = df.columns.tolist()
			primary_col = st.selectbox('Select Primary Column To Group By',all_columns_names)
			selected_column_names = st.multiselect('Select Columns',all_columns_names)
			if st.button("Plot"):
				st.text("Generating Plot for: {} and {}".format(primary_col,selected_column_names))
				if selected_column_names:
					vc_plot = df.groupby(primary_col)[selected_column_names].count()		
				else:
					vc_plot = df.iloc[:,-1].value_counts()
				st.write(vc_plot.plot(kind='bar'))
				st.pyplot()

		st.markdown("#### Pie Chart")
		if st.checkbox("Pie Plot"):
			all_columns_names = df.columns.tolist()
			# st.info("Please Choose Target Column")
			# int_column =  st.selectbox('Select Int Columns For Pie Plot',all_columns_names)
			if st.button("Generate Pie Plot"):
				# cust_values = df[int_column].value_counts()
				# st.write(cust_values.plot.pie(autopct="%1.1f%%"))
				st.write(df.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
				st.pyplot()
		


	# Choice Prediction
	if choice == 'Prediction':
		st.subheader("Prediction of Contraceptive Method Choice")
		st.markdown('<style>' + open('icon.css').read() + '</style>', unsafe_allow_html=True)
		st.markdown('<i class="material-icons">mood</i>', unsafe_allow_html=True)
		
		load_css('icon.css') # function defines at the top
		# load_icon('timeline') #function defines at the top	

		age = st.slider("Select Age",16,60)
		wife_education = st.number_input("Wife's Education Level(low2High) [1,4]",1,4)
		husband_education = st.number_input("Husband's Education Level(low2High) [1,4]",1,4)
		num_of_children_ever_born = st.number_input("Number of Children",value=0)

		wife_reg = {"Non_Religious":0,"Religious":1}
		choice_wife_reg = st.radio("Wife's Religion",tuple(wife_reg.keys()))
		result_wife_reg = get_value(choice_wife_reg,wife_reg)
		# st.text(result_wife_reg)


		wife_working = {"Yes":0,"No":1}
		choice_wife_working = st.radio("Is the Wife Currently Working",tuple(wife_working.keys()))
		result_wife_working = get_value(choice_wife_working,wife_working)
		# st.text(result_wife_working)


		husband_occupation = st.number_input("Husband Occupation(low2High) [1,4]",1,4)
		standard_of_living = st.slider("Standard of Living (low2High) [1,4]",1,4)

		media_exposure = {"Good":0,"Not Good":1}
		choice_media_exposure = st.radio("Media Exposure",tuple(media_exposure.keys()))
		result_media_exposure = get_value(choice_media_exposure,media_exposure)


		# Result and in json format
		results = [age,wife_education,husband_education,num_of_children_ever_born,result_wife_reg,result_wife_working,husband_occupation,standard_of_living,result_media_exposure]
		displayed_results = [age,wife_education,husband_education,num_of_children_ever_born,choice_wife_reg,choice_wife_working,husband_occupation,standard_of_living,choice_media_exposure]
		prettified_result = {"age":age,
		"wife_education":wife_education,
		"husband_education":husband_education,
		"num_of_children_ever_born":num_of_children_ever_born,
		"result_wife_reg":choice_wife_reg,
		"result_wife_working":choice_wife_working,
		"husband_occupation":husband_occupation,
		"standard_of_living":standard_of_living,
		"media_exposure":choice_media_exposure}
		sample_data = np.array(results).reshape(1, -1)
		
		
		if st.checkbox("Your Inputs Summary"):
			st.json(prettified_result)
			st.text("Vectorized as ::{}".format(results))

		st.subheader("Prediction")
		if st.checkbox("Make Prediction"):
			all_ml_dict = {'LR':LogisticRegression(),
			'RForest':RandomForestClassifier(),
			'MultNB':MultinomialNB()}
			# models = []
			# model_choice = st.multiselect('Model Choices',list(all_ml_dict.keys()))
			# for key in all_ml_dict:
			# 	if 'RForest' in key:
			# 		st.write(key)

			# Model Selection
			model_choice = st.selectbox('Model Choice',list(all_ml_dict.keys()))
			prediction_label = {"No-use": 1,"Long-term": 2,"Short-term":3}
			if st.button("Predict"):
				if model_choice == 'RForest':
					model_predictor = load_model("models/cmc_rf_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.text(prediction)
					# final_result = get_key(prediction,prediction_label)
					# st.info(final_result)
				elif model_choice == 'LR':
					model_predictor = load_model("models/cmc_logit_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.text(prediction)
				elif model_choice == 'MultNB':
					model_predictor = load_model("models/cmc_nv_model.pkl")
					prediction = model_predictor.predict(sample_data)
					# st.text(prediction)
				
				final_result = get_key(prediction,prediction_label)
				monitor = Monitor(age,wife_education,husband_education,num_of_children_ever_born,result_wife_reg,result_wife_working,husband_occupation,standard_of_living,result_media_exposure,final_result,model_choice)
				monitor.create_table()
				monitor.add_data()
				
				st.success(final_result)

				
	# CHOICE is Monitor
	if choice == 'Monitor':
		st.subheader("Metrics of Predictions")
		# Create your connection.
		conn = sqlite3.connect('data.db')

		fetched_data = pd.read_sql_query("SELECT * FROM cmcprediction", conn)
		st.dataframe(fetched_data)

	# ABOUT CHOICE
	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			#### Salary Predictor ML App
			##### Built with Streamlit

			#### By
			+ Hrishikesh Sharad Malkar
			+ References: Jesus Saves@[JCharisTech](https://jcharistech.com)

			""")


if __name__ == '__main__':
	main()
