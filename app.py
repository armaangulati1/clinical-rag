import os, requests, streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.title("💊 Clinical Drug-Safety Q&A")
st.caption("Answers grounded in FDA drug labels, with verifiable citations.")

q = st.text_input("Ask about metoprolol, lisinopril, metformin, atorvastatin, or warfarin:")
if st.button("Ask") and q.strip():
    with st.spinner("Searching the labels..."):
        resp = requests.post(f"{API_URL}/ask", json={"question": q}, timeout=120).json()
    st.markdown(resp["answer"])
    if resp["citations"]:
        st.subheader("Sources")
        for c in resp["citations"]:
            st.markdown(f"- **[{c['n']}] {c['drug']} — {c['section']}** — [view FDA label]({c['source']})")
            st.caption(c["snippet"] + "…")
    else:
        st.info("No sources cited — likely outside the loaded drug labels.")
