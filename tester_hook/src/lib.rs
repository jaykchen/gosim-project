use dotenv::dotenv;
use flowsnet_platform_sdk::logger;
use gosim_project::db_manipulate::*;
use gosim_project::db_populate::*;
use gosim_project::issue_tracker::*;
use gosim_project::llm_utils::chat_inner_async;
use gosim_project::vector_search::*;
use mysql_async::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use vector_store_flows::delete_collection;
use webhook_flows::{
    create_endpoint, request_handler,
    route::{get, post, route, RouteError, Router},
    send_response,
};

#[no_mangle]
#[tokio::main(flavor = "current_thread")]
pub async fn on_deploy() {
    create_endpoint().await;
}

#[request_handler(get, post)]
async fn handler(
    _headers: Vec<(String, String)>,
    _subpath: String,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    dotenv().ok();
    logger::init();

    let mut router = Router::new();
    router.insert("/run", vec![get(trigger)]).unwrap();
    router
        .insert("/deep", vec![post(check_deep_handler)])
        .unwrap();
    router
        .insert("/comment", vec![post(get_comments_by_post_handler)])
        .unwrap();
    router
        .insert("/vector", vec![post(check_vdb_by_post_handler)])
        .unwrap();
    router
        .insert("/vector/create", vec![post(create_vdb_handler)])
        .unwrap();
    router
        .insert("/vector/delete", vec![post(delete_vdb_handler)])
        .unwrap();

    if let Err(e) = route(router).await {
        match e {
            RouteError::NotFound => {
                send_response(404, vec![], b"No route matched".to_vec());
            }
            RouteError::MethodNotAllowed => {
                send_response(405, vec![], b"Method not allowed".to_vec());
            }
        }
    }
}

async fn get_comments_by_post_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct IssueId {
        pub issue_id: String,
    }

    let load: IssueId = match serde_json::from_slice(&_body) {
        Ok(obj) => obj,
        Err(_e) => {
            log::error!("failed to parse body: {}", _e);
            return;
        }
    };
    let pool: Pool = get_pool().await;

    let issue_id = load.issue_id;
    match get_comments_by_issue_id(&pool, &issue_id).await {
        Ok(result) => {
            let result_str = json!(result).to_string();

            send_response(
                200,
                vec![
                    (
                        String::from("content-type"),
                        String::from("application/json"),
                    ),
                    (
                        String::from("Access-Control-Allow-Origin"),
                        String::from("*"),
                    ),
                ],
                result_str.as_bytes().to_vec(),
            );
        }
        Err(e) => {
            log::error!("Error: {:?}", e);
        }
    }
}
async fn check_vdb_by_post_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct VectorLoad {
        pub issue_id: Option<String>,
        pub collection_name: Option<String>,
        pub text: Option<String>,
    }

    let load: VectorLoad = match serde_json::from_slice(&_body) {
        Ok(obj) => obj,
        Err(_e) => {
            log::error!("failed to parse body: {}", _e);
            return;
        }
    };
    if let Some(text) = load.text {
        match search_collection(&text, "gosim_search").await {
            Ok(search_result) => {
                send_response(
                    200,
                    vec![
                        (
                            String::from("content-type"),
                            String::from("application/json"),
                        ),
                        (
                            String::from("Access-Control-Allow-Origin"),
                            String::from("*"),
                        ),
                    ],
                    json!(search_result).to_string().as_bytes().to_vec(),
                );
            }
            Err(e) => {
                log::error!("Error: {:?}", e);
            }
        }
    }
    if let Some(collection_name) = load.collection_name {
        let result = check_vector_db(&collection_name).await;
        send_response(
            200,
            vec![
                (
                    String::from("content-type"),
                    String::from("application/json"),
                ),
                (
                    String::from("Access-Control-Allow-Origin"),
                    String::from("*"),
                ),
            ],
            json!(result).to_string().as_bytes().to_vec(),
        );
    }
}
async fn check_deep_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct VectorLoad {
        pub text: Option<String>,
    }
    let model = "meta-llama/Meta-Llama-3-8B-Instruct";

    if let Ok(load) = serde_json::from_slice::<VectorLoad>(&_body) {
        if let Some(text) = load.text {
            log::info!("text: {text}");
            if let Ok(reply) = chat_inner_async("you're an AI assistant", &text, 100, model).await {
                send_response(
                    200,
                    vec![
                        (
                            String::from("content-type"),
                            String::from("application/json"),
                        ),
                        (
                            String::from("Access-Control-Allow-Origin"),
                            String::from("*"),
                        ),
                    ],
                    json!(reply).to_string().as_bytes().to_vec(),
                );
            }
        }
    }
}

async fn delete_vdb_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct VectorLoad {
        pub collection_name: Option<String>,
    }

    let load: VectorLoad = match serde_json::from_slice(&_body) {
        Ok(obj) => obj,
        Err(_e) => {
            log::error!("failed to parse body: {}", _e);
            return;
        }
    };
    if let Some(collection_name) = load.collection_name {
        if let Err(e) = delete_collection(&collection_name).await {
            log::error!("Error deleting vector db: {:?}", e);
        }

        let result = check_vector_db(&collection_name).await;
        let out = json!(result).to_string();

        send_response(
            200,
            vec![
                (
                    String::from("content-type"),
                    String::from("application/json"),
                ),
                (
                    String::from("Access-Control-Allow-Origin"),
                    String::from("*"),
                ),
            ],
            out.as_bytes().to_vec(),
        );
    }
}
async fn create_vdb_handler(
    _headers: Vec<(String, String)>,
    _qry: HashMap<String, Value>,
    _body: Vec<u8>,
) {
    #[derive(Serialize, Deserialize, Clone, Debug, Default)]
    pub struct VectorLoad {
        pub collection_name: Option<String>,
    }

    let load: VectorLoad = match serde_json::from_slice(&_body) {
        Ok(obj) => obj,
        Err(_e) => {
            log::error!("failed to parse body: {}", _e);
            return;
        }
    };
    if let Some(collection_name) = load.collection_name {
        if let Err(e) = create_my_collection(1536, &collection_name).await {
            log::error!("Error creating vector db: {:?}", e);
        }

        let result = check_vector_db(&collection_name).await;
        let out = json!(result).to_string();

        send_response(
            200,
            vec![
                (
                    String::from("content-type"),
                    String::from("application/json"),
                ),
                (
                    String::from("Access-Control-Allow-Origin"),
                    String::from("*"),
                ),
            ],
            out.as_bytes().to_vec(),
        );
    }
}
async fn trigger(_headers: Vec<(String, String)>, _qry: HashMap<String, Value>, _body: Vec<u8>) {
    let pool: Pool = get_pool().await;
    // let _ = note_issues(&pool).await;

    // let repos = "repo:WasmEdge/wasmedge-db-examples repo:WasmEdge/www repo:WasmEdge/docs repo:WasmEdge/llvm-windows repo:WasmEdge/wasmedge-rust-sdk repo:WasmEdge/YOLO-rs repo:WasmEdge/proxy-wasm-cpp-host repo:WasmEdge/hyper-util repo:WasmEdge/hyper repo:WasmEdge/h2 repo:WasmEdge/wasmedge_hyper_demo repo:WasmEdge/tokio-rustls repo:WasmEdge/mysql_async_wasi repo:WasmEdge/mediapipe-rs repo:WasmEdge/wasmedge_reqwest_demo repo:WasmEdge/reqwest repo:WasmEdge/.github repo:WasmEdge/mio repo:WasmEdge/elasticsearch-rs-wasi repo:WasmEdge/oss-fuzz repo:WasmEdge/wasm-log-flex repo:WasmEdge/wasmedge_sdk_async_wasi repo:WasmEdge/tokio repo:WasmEdge/rust-mysql-simple-wasi repo:WasmEdge/GSoD2023 repo:WasmEdge/llm-agent-sdk repo:WasmEdge/sqlx repo:WasmEdge/rust-postgres repo:WasmEdge/redis-rs";

    let query_repos: String = get_projects_as_repo_list(&pool, 1).await.expect("failed to get projects as repo list");

    let repo_data_vec: Vec<RepoData> = search_repos_in_batch(&query_repos).await.expect("failed to search repos data");

    for repo_data in repo_data_vec {
        let _ = fill_project_w_repo_data(&pool, repo_data.clone()).await.expect("failed to fill projects table");
        let _ = summarize_project_add_in_db_one_step(&pool, repo_data).await.expect("failed to summarize and mark in db");
    }
}
