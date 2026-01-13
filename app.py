import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Spark Results Viewer", layout="wide")

st.title("Cloud Data Processing Results")
st.write("View results from Google Colab Spark processing")

st.divider()

resultsfile = st.file_uploader("Upload results.json from Google Drive", type=["json"])

if resultsfile is not None:
    try:
        content = resultsfile.read().decode("utf-8")
        results = json.loads(content)
        
        st.success("File loaded")
        
        st.divider()
        st.header("Dataset")
        c1, c2, c3 = st.columns(3)
        c1.metric("File", results["dataset"]["filename"])
        c2.metric("Rows", results["dataset"]["rows"])
        c3.metric("Columns", results["dataset"]["columns"])
        
        st.divider()
        st.header("Statistics")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Rows", results["statistics"]["numrows"])
        s2.metric("Total Columns", results["statistics"]["numcols"])
        s3.metric("Numeric Columns", results["statistics"]["numericcols"])
        s4.metric("Clean Rows", results["statistics"]["cleanrows"])
        
        st.divider()
        st.header("Machine Learning")
        
        ml = results["mlresults"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Linear Regression")
            lr = ml["linearregression"]
            st.write("RMSE: " + str(lr["rmse"]))
            st.write("R2: " + str(lr["r2"]))
            st.write("Time: " + str(lr["time"]) + " sec")
            
            st.subheader("Random Forest")
            rf = ml["randomforest"]
            st.write("RMSE: " + str(rf["rmse"]))
            st.write("R2: " + str(rf["r2"]))
            st.write("Trees: " + str(rf["trees"]))
        
        with col2:
            st.subheader("KMeans")
            km = ml["kmeans"]
            st.write("WSSSE: " + str(km["wssse"]))
            st.write("Clusters: " + str(km["clusters"]))
            st.write("Time: " + str(km["time"]) + " sec")
            
            st.subheader("Correlation")
            corr = ml["correlation"]
            st.write("Strongest: " + str(corr["strongest"]))
            st.write("Value: " + str(corr["value"]))
        
        st.divider()
        st.header("Benchmark")
        
        bench = results["benchmark"]
        benchdf = pd.DataFrame(bench)
        benchdf.columns = ["Cores", "Time", "Speedup", "Efficiency"]
        st.dataframe(benchdf, hide_index=True)
        
        st.bar_chart(benchdf.set_index("Cores")["Speedup"])
        
        st.divider()
        st.header("Options Used")
        opt = results["options"]
        o1, o2, o3 = st.columns(3)
        o1.metric("Clusters", opt["clusters"])
        o2.metric("Trees", opt["trees"])
        o3.metric("Test Split", str(opt["testsplit"]) + "%")
        
    except Exception as e:
        st.error("Error: " + str(e))
        st.write("Make sure you uploaded the correct results.json file")

else:
    st.info("Upload results.json to view results")
    st.write("Steps:")
    st.write("1. Run Colab notebook")
    st.write("2. Download results.json from Google Drive sparkresults folder")
    st.write("3. Upload here")
