defmodule AiDataFoundationWeb.PageControllerTest do
  use AiDataFoundationWeb.ConnCase

  test "GET / renders home page", %{conn: conn} do
    conn = get(conn, ~p"/")
    assert html_response(conn, 200) =~ "Defend your AI"
    assert html_response(conn, 200) =~ "Featured Security Labs"
    assert html_response(conn, 200) =~ "Open Weight Model Benchmarks"
    assert html_response(conn, 200) =~ "Qwen 32B"
    assert html_response(conn, 200) =~ "76.0%"
    assert html_response(conn, 200) =~ "LF AI &amp; Data Foundation Projects"
    assert html_response(conn, 200) =~ "Graduated Stage"
  end

  test "GET /tools renders tools catalog container", %{conn: conn} do
    conn = get(conn, ~p"/tools")
    assert html_response(conn, 200) =~ "modern AI systems"
    assert html_response(conn, 200) =~ "kube-tools"
    assert html_response(conn, 200) =~ "LF AI &amp; Data Foundation"
  end

  test "GET /models renders open model leaderboard, VRAM calculator, and agents", %{conn: conn} do
    conn = get(conn, ~p"/models")
    assert html_response(conn, 200) =~ "VRAM Calculator"
    assert html_response(conn, 200) =~ "Qwen 32B"
    assert html_response(conn, 200) =~ "DeepSeek V3"
    assert html_response(conn, 200) =~ "OpenClaw"
    assert html_response(conn, 200) =~ "vram-params"
  end

  test "GET /labs and GET /labs/:id", %{conn: conn} do
    conn_list = get(conn, ~p"/labs")
    assert html_response(conn_list, 200) =~ "Interactive Practice"
    assert html_response(conn_list, 200) =~ "LangChain Practical Labs"

    conn_detail = get(conn, ~p"/labs/LangChain-Practical-Labs")
    assert html_response(conn_detail, 200) =~ "LangChain Practical Labs"
    assert html_response(conn_detail, 200) =~ "Interactive Runbook"

    conn_mpl = get(conn, ~p"/labs/matplotlib-complete-guide")
    assert html_response(conn_mpl, 200) =~ "Matplotlib Complete Guide"
    assert html_response(conn_mpl, 200) =~ "Interactive Runbook"

    conn_sns = get(conn, ~p"/labs/seaborn-complete-guide")
    assert html_response(conn_sns, 200) =~ "Seaborn Complete Guide"
    assert html_response(conn_sns, 200) =~ "Interactive Runbook"

    conn_skl = get(conn, ~p"/labs/scikit-learn-complete-guide")
    assert html_response(conn_skl, 200) =~ "Scikit-learn Complete Guide"
    assert html_response(conn_skl, 200) =~ "Interactive Runbook"

    conn_xgb = get(conn, ~p"/labs/xgboost-complete-guide")
    assert html_response(conn_xgb, 200) =~ "XGBoost Complete Guide"
    assert html_response(conn_xgb, 200) =~ "Interactive Runbook"

    conn_lgb = get(conn, ~p"/labs/lightgbm-complete-guide")
    assert html_response(conn_lgb, 200) =~ "LightGBM Complete Learning Guide"
    assert html_response(conn_lgb, 200) =~ "Interactive Runbook"

    conn_cat = get(conn, ~p"/labs/catboost-complete-guide")
    assert html_response(conn_cat, 200) =~ "CatBoost Complete Guide"
    assert html_response(conn_cat, 200) =~ "Interactive Runbook"

    conn_pil = get(conn, ~p"/labs/pillow-complete-guide")
    assert html_response(conn_pil, 200) =~ "Complete Pillow Guide"
    assert html_response(conn_pil, 200) =~ "Interactive Runbook"

    conn_cv = get(conn, ~p"/labs/opencv-complete-guide")
    assert html_response(conn_cv, 200) =~ "OpenCV Python Complete Guide"
    assert html_response(conn_cv, 200) =~ "Interactive Runbook"

    conn_ski = get(conn, ~p"/labs/scikit-image-complete-guide")
    assert html_response(conn_ski, 200) =~ "scikit-image Complete Guide"
    assert html_response(conn_ski, 200) =~ "Interactive Runbook"

    conn_torch = get(conn, ~p"/labs/pytorch-complete-guide")
    assert html_response(conn_torch, 200) =~ "PyTorch Complete Guide"
    assert html_response(conn_torch, 200) =~ "Interactive Runbook"

    conn_tv = get(conn, ~p"/labs/torchvision-complete-guide")
    assert html_response(conn_tv, 200) =~ "TorchVision Complete Guide"
    assert html_response(conn_tv, 200) =~ "Interactive Runbook"

    conn_tf = get(conn, ~p"/labs/tensorflow-complete-guide")
    assert html_response(conn_tf, 200) =~ "TensorFlow Complete Guide"
    assert html_response(conn_tf, 200) =~ "Interactive Runbook"

    conn_keras = get(conn, ~p"/labs/keras-complete-guide")
    assert html_response(conn_keras, 200) =~ "Keras 3 Complete Guide"
    assert html_response(conn_keras, 200) =~ "Interactive Runbook"
  end

  test "GET /blog and GET /blog/:id", %{conn: conn} do
    conn_list = get(conn, ~p"/blog")
    assert html_response(conn_list, 200) =~ "The ADF Journal"
    assert html_response(conn_list, 200) =~ "Qwen Coder Models"

    conn_detail = get(conn, ~p"/blog/Qwen-Coder-Models")
    assert html_response(conn_detail, 200) =~ "Qwen Coder Models"
    assert html_response(conn_detail, 200) =~ "Sangam Biradar"
  end

  test "GET /roadmap renders editorial tracks", %{conn: conn} do
    conn = get(conn, ~p"/roadmap")
    assert html_response(conn, 200) =~ "1,000 practical titles"
    assert html_response(conn, 200) =~ "LLM Architectures &amp; Foundations"
    assert html_response(conn, 200) =~ "Agentic Systems &amp; Model Context Protocol"
  end

  test "GET /about renders mission and pillars", %{conn: conn} do
    conn = get(conn, ~p"/about")
    assert html_response(conn, 200) =~ "Democratizing Artificial Intelligence"
    assert html_response(conn, 200) =~ "Open Protocols & Tooling"
    assert html_response(conn, 200) =~ "Project Maturity Lifecycle"
  end

  test "GET /sitemap.xml and GET /rss.xml", %{conn: conn} do
    conn_sitemap = get(conn, ~p"/sitemap.xml")
    assert response_content_type(conn_sitemap, :xml) =~ "xml"
    assert conn_sitemap.resp_body =~ "<urlset"

    conn_rss = get(conn, ~p"/rss.xml")
    assert response_content_type(conn_rss, :xml) =~ "xml"
    assert conn_rss.resp_body =~ "<rss version=\"2.0\""
  end

  test "returns 404 for unknown lab and post", %{conn: conn} do
    conn_lab = get(conn, ~p"/labs/non-existent-lab-id")
    assert html_response(conn_lab, 404) =~ "Not Found"

    conn_post = get(conn, ~p"/blog/non-existent-post-id")
    assert html_response(conn_post, 404) =~ "Not Found"
  end
end
