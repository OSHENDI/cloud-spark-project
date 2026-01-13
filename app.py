import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Spark Results Viewer", layout="wide")

st.title("Cloud Spark Results Viewer")
st.caption("Upload your results.json file to view ML analytics")

resultsfile = st.file_uploader("Upload results.json", type=["json"])

if resultsfile is not None:
    try:
        results = json.loads(resultsfile.read().decode("utf-8"))
        st.success("Results loaded successfully")
        
        st.header("Dataset Info")
        c1, c2, c3 = st.columns(3)
        c1.metric("File", results["dataset"]["filename"])
        c2.metric("Rows", f'{results["dataset"]["rows"]:,}')
        c3.metric("Columns", results["dataset"]["columns"])
        
        st.header("Machine Learning Results")
        
        ml = results["mlresults"]
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Linear Regression")
            st.metric("RMSE", ml["linearregression"]["rmse"])
            st.metric("R2 Score", ml["linearregression"]["r2"])
            
            st.subheader("Random Forest")
            st.metric("RMSE", ml["randomforest"]["rmse"])
            st.metric("R2 Score", ml["randomforest"]["r2"])
        
        with col2:
            st.subheader("KMeans Clustering")
            st.metric("Clusters", ml["kmeans"]["clusters"])
            st.metric("WSSSE", f'{ml["kmeans"]["wssse"]:,.0f}')
            
            st.subheader("Correlation Analysis")
            st.metric("Strongest Predictor", ml["correlation"]["strongest"])
            st.metric("Correlation Value", ml["correlation"]["value"])
        
        st.header("Scalability Benchmark")
        bench = results["benchmark"]
        benchdf = pd.DataFrame(bench)
        benchdf.columns = ["Cores", "Time (sec)", "Speedup", "Efficiency (%)"]
        st.dataframe(benchdf, hide_index=True, use_container_width=True)
        st.bar_chart(benchdf.set_index("Cores")["Speedup"])
        
    except Exception as e:
        st.error("Error loading file: " + str(e))

else:
    st.info("Upload your results.json file from Google Drive")
