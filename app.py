import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Employee Attrition Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-label {
        font-size: 1rem;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('employee_attrition_data.csv')
    return df

@st.cache_data
def preprocess_data(df):
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Convert Attrition to categorical
    data['Attrition_Label'] = data['Attrition'].map({0: 'No', 1: 'Yes'})
    
    # Create age groups
    data['Age_Group'] = pd.cut(data['Age'], 
                                bins=[20, 30, 40, 50, 60], 
                                labels=['20-30', '30-40', '40-50', '50-60'])
    
    # Create satisfaction level groups
    data['Satisfaction_Level_Group'] = pd.cut(data['Satisfaction_Level'],
                                               bins=[0, 0.25, 0.5, 0.75, 1.0],
                                               labels=['Very Low', 'Low', 'Medium', 'High'])
    
    # Create salary groups
    data['Salary_Group'] = pd.cut(data['Salary'],
                                   bins=[0, 40000, 60000, 80000, 100000],
                                   labels=['<40K', '40K-60K', '60K-80K', '80K-100K'])
    
    return data

# Load data
try:
    df = load_data()
    data = preprocess_data(df)
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/group-of-projects.png", width=100)
    st.sidebar.title("Navigation")
    
    pages = ["📊 Overview", "🔍 Detailed Analysis", "📈 Predictive Modeling", "ℹ️ About"]
    selected_page = st.sidebar.radio("Go to", pages)
    
    # Filters
    st.sidebar.header("Filters")
    
    departments = st.sidebar.multiselect(
        "Select Departments",
        options=data['Department'].unique(),
        default=data['Department'].unique()
    )
    
    gender = st.sidebar.multiselect(
        "Select Gender",
        options=data['Gender'].unique(),
        default=data['Gender'].unique()
    )
    
    attrition_filter = st.sidebar.multiselect(
        "Attrition Status",
        options=['Yes', 'No'],
        default=['Yes', 'No']
    )
    
    # Apply filters
    filtered_data = data[
        (data['Department'].isin(departments)) &
        (data['Gender'].isin(gender)) &
        (data['Attrition_Label'].isin(attrition_filter))
    ]
    
    # Main content based on selected page
    if selected_page == "📊 Overview":
        st.markdown("<h1 class='main-header'>Employee Attrition Dashboard</h1>", unsafe_allow_html=True)
        
        # Key Metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_employees = len(filtered_data)
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{total_employees:,}</div>
                <div class='metric-label'>Total Employees</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            attrition_count = len(filtered_data[filtered_data['Attrition'] == 1])
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{attrition_count:,}</div>
                <div class='metric-label'>Attrition Count</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            attrition_rate = (attrition_count / total_employees * 100) if total_employees > 0 else 0
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{attrition_rate:.1f}%</div>
                <div class='metric-label'>Attrition Rate</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            avg_satisfaction = filtered_data['Satisfaction_Level'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{avg_satisfaction:.2f}</div>
                <div class='metric-label'>Avg Satisfaction</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            avg_salary = filtered_data['Salary'].mean()
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>${avg_salary:,.0f}</div>
                <div class='metric-label'>Avg Salary</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Attrition by Department
            st.markdown("<h3 class='sub-header'>Attrition by Department</h3>", unsafe_allow_html=True)
            dept_attrition = filtered_data.groupby('Department')['Attrition'].agg(['count', 'sum'])
            dept_attrition['rate'] = (dept_attrition['sum'] / dept_attrition['count'] * 100)
            dept_attrition = dept_attrition.reset_index()
            
            fig = px.bar(dept_attrition, x='rate', y='Department', 
                         color='rate', color_continuous_scale='Reds',
                         labels={'rate': 'Attrition Rate (%)'},
                         title='Attrition Rate by Department')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Gender distribution
            st.markdown("<h3 class='sub-header'>Gender Distribution</h3>", unsafe_allow_html=True)
            gender_counts = filtered_data['Gender'].value_counts().reset_index()
            gender_counts.columns = ['Gender', 'Count']
            
            fig = px.pie(gender_counts, values='Count', names='Gender', 
                         title='Gender Distribution',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Attrition by Age Group
            st.markdown("<h3 class='sub-header'>Attrition by Age Group</h3>", unsafe_allow_html=True)
            age_attrition = filtered_data.groupby('Age_Group', observed=True)['Attrition'].agg(['count', 'sum'])
            age_attrition['rate'] = (age_attrition['sum'] / age_attrition['count'] * 100)
            age_attrition = age_attrition.reset_index()
            
            fig = px.line(age_attrition, x='rate', y='Age_Group', 
                          markers=True, title='Attrition Rate by Age Group',
                          labels={'rate': 'Attrition Rate (%)', 'Age_Group': 'Age Group'})
            fig.update_traces(line_color='red', line_width=3)
            st.plotly_chart(fig, use_container_width=True)
            
            # Attrition by Promotion
            st.markdown("<h3 class='sub-header'>Attrition by Promotion Status</h3>", unsafe_allow_html=True)
            promo_attrition = filtered_data.groupby('Promotion_Last_5Years')['Attrition'].agg(['count', 'sum'])
            promo_attrition['rate'] = (promo_attrition['sum'] / promo_attrition['count'] * 100)
            promo_attrition = promo_attrition.reset_index()
            promo_attrition['Promotion_Last_5Years'] = promo_attrition['Promotion_Last_5Years'].map({0: 'No Promotion', 1: 'Promoted'})
            
            fig = px.bar(promo_attrition, x='rate', y='Promotion_Last_5Years',
                         color='rate', color_continuous_scale='Viridis',
                         labels={'rate': 'Attrition Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Satisfaction vs Attrition
        st.markdown("<h3 class='sub-header'>Satisfaction Level Distribution</h3>", unsafe_allow_html=True)
        fig = px.histogram(filtered_data, x='Satisfaction_Level', color='Attrition_Label',
                           nbins=50, barmode='overlay',
                           labels={'Satisfaction_Level': 'Satisfaction Level', 'count': 'Count'},
                           color_discrete_map={'Yes': 'red', 'No': 'blue'})
        fig.update_layout(legend_title_text='Attrition')
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Heatmap
        st.markdown("<h3 class='sub-header'>Correlation Heatmap</h3>", unsafe_allow_html=True)
        numeric_cols = ['Age', 'Years_at_Company', 'Satisfaction_Level', 
                        'Average_Monthly_Hours', 'Promotion_Last_5Years', 'Salary', 'Attrition']
        corr_matrix = filtered_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        color_continuous_scale='RdBu_r',
                        title='Feature Correlations')
        st.plotly_chart(fig, use_container_width=True)
    
    elif selected_page == "🔍 Detailed Analysis":
        st.markdown("<h1 class='main-header'>Detailed Attrition Analysis</h1>", unsafe_allow_html=True)
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["Demographics", "Job Factors", "Satisfaction & Hours", "Salary Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution by attrition
                fig = px.box(filtered_data, x='Attrition_Label', y='Age', 
                             color='Attrition_Label',
                             title='Age Distribution by Attrition Status',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Gender and Attrition
                gender_attrition = pd.crosstab(filtered_data['Gender'], filtered_data['Attrition_Label'], normalize='index') * 100
                gender_attrition = gender_attrition.reset_index().melt(id_vars=['Gender'], var_name='Attrition', value_name='Percentage')
                
                fig = px.bar(gender_attrition, x='Gender', y='Percentage', color='Attrition',
                             title='Attrition Percentage by Gender',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'},
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Age groups detailed
                age_group_attrition = pd.crosstab(filtered_data['Age_Group'], filtered_data['Attrition_Label'], normalize='index') * 100
                age_group_attrition = age_group_attrition.reset_index().melt(id_vars=['Age_Group'], var_name='Attrition', value_name='Percentage')
                
                fig = px.bar(age_group_attrition, x='Age_Group', y='Percentage', color='Attrition',
                             title='Attrition Percentage by Age Group',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'},
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Department vs Job Title heatmap
                dept_job_attrition = filtered_data.groupby(['Department', 'Job_Title'])['Attrition'].mean().reset_index()
                dept_job_attrition['Attrition'] = dept_job_attrition['Attrition'] * 100
                
                fig = px.density_heatmap(dept_job_attrition, x='Department', y='Job_Title', z='Attrition',
                                         color_continuous_scale='Reds',
                                         title='Attrition Rate by Department and Job Title (%)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Years at Company analysis
                fig = px.violin(filtered_data, x='Attrition_Label', y='Years_at_Company',
                                color='Attrition_Label', box=True,
                                title='Years at Company Distribution by Attrition',
                                color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Promotion impact
            st.markdown("### Promotion Impact on Attrition")
            promo_stats = filtered_data.groupby(['Promotion_Last_5Years', 'Attrition_Label']).size().reset_index(name='Count')
            promo_stats['Promotion_Last_5Years'] = promo_stats['Promotion_Last_5Years'].map({0: 'No Promotion', 1: 'Promoted'})
            
            fig = px.bar(promo_stats, x='Promotion_Last_5Years', y='Count', color='Attrition_Label',
                         title='Attrition Count by Promotion Status',
                         color_discrete_map={'Yes': 'red', 'No': 'blue'},
                         barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Satisfaction Level by Attrition
                fig = px.box(filtered_data, x='Attrition_Label', y='Satisfaction_Level',
                             color='Attrition_Label',
                             title='Satisfaction Level Distribution by Attrition',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Satisfaction Level Groups
                sat_attrition = pd.crosstab(filtered_data['Satisfaction_Level_Group'], filtered_data['Attrition_Label'], normalize='index') * 100
                sat_attrition = sat_attrition.reset_index().melt(id_vars=['Satisfaction_Level_Group'], var_name='Attrition', value_name='Percentage')
                
                fig = px.bar(sat_attrition, x='Satisfaction_Level_Group', y='Percentage', color='Attrition',
                             title='Attrition Rate by Satisfaction Level',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'},
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average Monthly Hours
                fig = px.box(filtered_data, x='Attrition_Label', y='Average_Monthly_Hours',
                             color='Attrition_Label',
                             title='Average Monthly Hours by Attrition',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
                
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                # Salary distribution
                fig = px.histogram(filtered_data, x='Salary', color='Attrition_Label',
                                   nbins=50, title='Salary Distribution by Attrition',
                                   color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Salary by Department
                fig = px.box(filtered_data, x='Department', y='Salary', color='Attrition_Label',
                             title='Salary Distribution by Department',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Salary Groups
                salary_attrition = pd.crosstab(filtered_data['Salary_Group'], filtered_data['Attrition_Label'], normalize='index') * 100
                salary_attrition = salary_attrition.reset_index().melt(id_vars=['Salary_Group'], var_name='Attrition', value_name='Percentage')
                
                fig = px.bar(salary_attrition, x='Salary_Group', y='Percentage', color='Attrition',
                             title='Attrition Rate by Salary Group',
                             color_discrete_map={'Yes': 'red', 'No': 'blue'},
                             barmode='group')
                st.plotly_chart(fig, use_container_width=True)
    
    
    elif selected_page == "📈 Predictive Modeling":
        st.markdown("<h1 class='main-header'>Attrition Prediction Model</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        This section builds a machine learning model to predict employee attrition based on various features.
        """)
        
        # Prepare data for modeling
        model_data = data.copy()
        
        # Encode categorical variables
        le_department = LabelEncoder()
        le_gender = LabelEncoder()
        le_job = LabelEncoder()
        
        model_data['Department_Encoded'] = le_department.fit_transform(model_data['Department'])
        model_data['Gender_Encoded'] = le_gender.fit_transform(model_data['Gender'])
        model_data['Job_Title_Encoded'] = le_job.fit_transform(model_data['Job_Title'])
        
        # Features for modeling
        features = ['Age', 'Years_at_Company', 'Satisfaction_Level', 'Average_Monthly_Hours',
                    'Promotion_Last_5Years', 'Salary', 'Department_Encoded', 'Gender_Encoded',
                    'Job_Title_Encoded']
        
        X = model_data[features]
        y = model_data['Attrition']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Model Configuration")
            n_estimators = st.slider("Number of Trees", min_value=50, max_value=300, value=100, step=50)
            max_depth = st.slider("Max Depth", min_value=3, max_value=20, value=10)
            
            if st.button("Train Model", type="primary"):
                # Train Random Forest
                rf_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                rf_model.fit(X_train, y_train)
                
                # Predictions
                y_pred = rf_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                st.session_state['model'] = rf_model
                st.session_state['accuracy'] = accuracy
                st.session_state['features'] = features
                st.session_state['trained'] = True
                
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
        
        with col2:
            if 'trained' in st.session_state and st.session_state['trained']:
                st.markdown("### Model Performance")
                st.metric("Accuracy", f"{st.session_state['accuracy']:.2%}")
                
                # Feature Importance
                importance = pd.DataFrame({
                    'Feature': features,
                    'Importance': st.session_state['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                             title='Feature Importance',
                             color='Importance', color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        
        # Prediction Section
        st.markdown("---")
        st.markdown("### Make a Prediction")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_age = st.number_input("Age", min_value=18, max_value=70, value=35)
            pred_years = st.number_input("Years at Company", min_value=0, max_value=40, value=5)
            pred_satisfaction = st.slider("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        
        with col2:
            pred_hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=200)
            pred_promotion = st.selectbox("Promotion in Last 5 Years", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            pred_salary = st.number_input("Salary", min_value=20000, max_value=200000, value=60000)
        
        with col3:
            pred_dept = st.selectbox("Department", model_data['Department'].unique())
            pred_gender = st.selectbox("Gender", model_data['Gender'].unique())
            pred_job = st.selectbox("Job Title", model_data['Job_Title'].unique())
        
        if st.button("Predict Attrition", type="primary"):
            if 'trained' in st.session_state and st.session_state['trained']:
                # Encode inputs
                dept_encoded = le_department.transform([pred_dept])[0]
                gender_encoded = le_gender.transform([pred_gender])[0]
                job_encoded = le_job.transform([pred_job])[0]
                
                # Create prediction array
                pred_array = np.array([[pred_age, pred_years, pred_satisfaction, pred_hours,
                                        pred_promotion, pred_salary, dept_encoded, gender_encoded, job_encoded]])
                
                # Make prediction
                prediction = st.session_state['model'].predict(pred_array)[0]
                probability = st.session_state['model'].predict_proba(pred_array)[0]
                
                if prediction == 1:
                    st.error(f"⚠️ High Risk of Attrition (Probability: {probability[1]:.2%})")
                else:
                    st.success(f"✅ Low Risk of Attrition (Probability: {probability[0]:.2%})")
            else:
                st.warning("Please train the model first!")
    
    else:  # About page
        st.markdown("<h1 class='main-header'>About This Dashboard</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        ## Employee Attrition Analytics Dashboard
        
        This interactive dashboard provides comprehensive analytics for understanding and predicting employee attrition.
        
        ### Features:
        - **Overview Dashboard**: Key metrics and high-level visualizations
        - **Detailed Analysis**: Deep dive into demographics, job factors, satisfaction, and salary
        - **Predictive Modeling**: Machine learning model to predict attrition risk
        
        ### Dataset Information:
        - **Total Records**: 1,000 employees
        - **Features**: Age, Gender, Department, Job Title, Years at Company, 
                       Satisfaction Level, Average Monthly Hours, Promotion History, Salary
        - **Target Variable**: Attrition (0 = No, 1 = Yes)
        
        ### Key Insights:
        - Lower satisfaction levels correlate with higher attrition
        - Salary and promotion history impact retention
        - Certain departments and job titles show different attrition patterns
        
        ### How to Use:
        1. Use the sidebar filters to focus on specific segments
        2. Navigate between pages using the radio buttons
        3. In the Predictive Modeling page, train the model and make predictions
        """)
        
        # Sample data
        st.markdown("### Sample Data")
        st.dataframe(data.head(10))
        
        # Data Summary
        st.markdown("### Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Numerical Features**")
            st.dataframe(data.describe())
        
        with col2:
            st.markdown("**Categorical Features**")
            cat_cols = ['Department', 'Gender', 'Job_Title']
            for col in cat_cols:
                st.markdown(f"**{col}**")
                st.dataframe(data[col].value_counts().reset_index())

except FileNotFoundError:
    st.error("""
    ⚠️ File 'employee_attrition_data.csv' not found!
    
    Please make sure the CSV file is in the same directory as this script.
    """)
    
    # Sample data creation option
    if st.button("Create Sample Data"):
        # Create sample data with similar structure
        np.random.seed(42)
        n_samples = 100
        
        sample_data = pd.DataFrame({
            'Employee_ID': range(n_samples),
            'Age': np.random.randint(22, 60, n_samples),
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Department': np.random.choice(['Sales', 'Engineering', 'HR', 'Marketing', 'Finance'], n_samples),
            'Job_Title': np.random.choice(['Manager', 'Engineer', 'Analyst', 'HR Specialist', 'Accountant'], n_samples),
            'Years_at_Company': np.random.randint(1, 15, n_samples),
            'Satisfaction_Level': np.random.uniform(0, 1, n_samples),
            'Average_Monthly_Hours': np.random.randint(120, 280, n_samples),
            'Promotion_Last_5Years': np.random.choice([0, 1], n_samples),
            'Salary': np.random.randint(30000, 100000, n_samples),
            'Attrition': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        sample_data.to_csv('employee_attrition_data.csv', index=False)
        st.success("Sample data created! Please refresh the page.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("👥 Employee Attrition Dashboard | Created with Streamlit") 