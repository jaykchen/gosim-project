use openai_flows::{embeddings::EmbeddingsInput, OpenAIFlows};
use regex::Regex;
use serde_json::json;
use std::env;
use vector_store_flows::*;

pub async fn upload_to_collection(
    issue_or_project_id: &str,
    content: String,
) -> anyhow::Result<()> {
    let collection_name = env::var("collection_name").unwrap_or("gosim_search".to_string());

    let id: u64 = match collection_info(&collection_name).await {
        Ok(ci) => ci.points_count,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Cannot get collection, can not init points_count: {}",
                e
            ))
        }
    };

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(2);

    let input = EmbeddingsInput::String(content.clone());
    match openai.create_embeddings(input).await {
        Ok(r) => {
            let v = &r[0];
            let p = vec![Point {
                id: PointId::Num(id),
                vector: v.iter().map(|n| *n as f32).collect(),
                payload: json!({
                        "issue_or_project_id": issue_or_project_id,
                        "text": content})
                .as_object()
                .map(|m| m.to_owned()),
            }];

            if let Err(e) = upsert_points(&collection_name, p).await {
                log::error!("Cannot upsert into database! {}", e);
            }
            log::debug!(
                "Created vector {} with length {}",
                issue_or_project_id,
                v.len()
            );

            Ok(())
        }
        Err(e) => {
            log::error!("OpenAI returned an error: {}", e);
            Err(anyhow::anyhow!("OpenAI returned an error: {}", e))
        }
    }
}

/* pub async fn upload_to_collection(
    issue_or_project_id: &str,
    content: String,
) -> anyhow::Result<()> {
    let collection_name = env::var("collection_name").unwrap_or("gosim_search".to_string());

    let mut id: u64 = match collection_info(&collection_name).await {
        Ok(ci) => ci.points_count,
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Cannot get collection, can not init points_count: {}",
                e
            ))
        }
    };

    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let input = EmbeddingsInput::String(content.clone());
    match openai.create_embeddings(input).await {
        Ok(r) => {
            for v in r.iter() {
                let p = vec![Point {
                    id: PointId::Num(id),
                    vector: v.iter().map(|n| *n as f32).collect(),
                    payload: json!({
                        "issue_or_project_id": issue_or_project_id,
                        "text": content})
                    .as_object()
                    .map(|m| m.to_owned()),
                }];

                if let Err(e) = upsert_points(&collection_name, p).await {
                    log::error!("Cannot upsert into database! {}", e);
                    return Ok(());
                }
                id += 1;
                log::debug!(
                    "Created vector {} with length {}",
                    issue_or_project_id,
                    v.len()
                );
            }
            Ok(())
        }
        Err(e) => {
            log::error!("OpenAI returned an error: {}", e);
            Err(anyhow::anyhow!("OpenAI returned an error: {}", e))
        }
    }
} */

pub async fn check_vector_db(collection_name: &str) -> String {
    match collection_info(collection_name).await {
        Ok(ci) => {
            log::info!(
                "The collection now has {} records in total.",
                ci.points_count
            );
            format!(
                "The collection now has {} records in total.",
                ci.points_count
            )
        }
        Err(e) => {
            log::error!("Cannot get collection: {} Error: {}", collection_name, e);
            format!("Cannot get collection: {} Error: {}", collection_name, e)
        }
    }
}

use std::cmp::Reverse;

pub async fn search_collection_hybrid(
    question: &str,
    collection_name: &str,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let project_regex = Regex::new(r"\bproject\b")?;
    let issue_regex = Regex::new(r"\bissue\b")?;

    let is_project = project_regex.is_match(&question.to_ascii_lowercase());
    let is_issue = issue_regex.is_match(&question.to_ascii_lowercase());

    let mut project_vec = Vec::new();
    let mut issue_vec = Vec::new();

    let question_vector = match openai
        .create_embeddings(EmbeddingsInput::String(question.to_string()))
        .await
    {
        Ok(r) if !r.is_empty() => r[0].iter().map(|n| *n as f32).collect(),
        _ => {
            log::error!("Failed to get embeddings for the question");
            return Err(anyhow::anyhow!("Failed to get embeddings for the question"));
        }
    };

    let p = PointsSearchParams {
        vector: question_vector,
        limit: 10,
    };

    let search_results = search_points(collection_name, &p).await.expect("search point failure");
    for p in search_results.iter() {
        let p_text = p
            .payload
            .as_ref()
            .unwrap()
            .get("text")
            .unwrap()
            .as_str()
            .unwrap();

        let issue_or_project_id = p
            .payload
            .as_ref()
            .unwrap()
            .get("issue_or_project_id")
            .unwrap()
            .as_str()
            .unwrap();
        let is_sid = issue_or_project_id.split('/').count() == 7;

        if p.score > 0.75 {
            let entry = (
                Reverse(p.score),
                issue_or_project_id.to_string(),
                p_text.to_string(),
            );
            if is_sid {
                issue_vec.push(entry);
            } else {
                project_vec.push(entry);
            }
        }
    }

    project_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    issue_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut results = Vec::new();
    let desired_result_count = 5;

    // Function to extract data from sorted vectors
    fn extract_results(
        vec: &mut Vec<(Reverse<f32>, String, String)>,
        results: &mut Vec<(String, String)>,
        count: usize,
    ) {
        while let Some((_score, id, text)) = vec.pop() {
            results.push((id, text));
            if results.len() >= count {
                break;
            }
        }
    }

    if is_project {
        extract_results(&mut project_vec, &mut results, desired_result_count);
    }
    if is_issue && results.len() < desired_result_count {
        extract_results(&mut issue_vec, &mut results, desired_result_count);
    }

    // If neither category alone provides enough results, combine them.
    if results.len() < desired_result_count {
        extract_results(&mut project_vec, &mut results, desired_result_count);
        extract_results(&mut issue_vec, &mut results, desired_result_count);
    }

    Ok(results)
}

// some logic that filters issue_vec and project_vec, combine the filtered result and output Vec<(String, String)>
// if the query intends to search projects, if the project_vec has enough candidates, i.e. > 3, use their values as output
//by same token, if the query intends to search issues, ...
// if the query intends to search projects, but project_vec is less than 3, take top scored from issues_heap, make the output <=5
// similarly, if the query intends to search projects, ...

pub async fn search_collection(
    question: &str,
    collection_name: &str,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let question_vector = match openai
        .create_embeddings(EmbeddingsInput::String(question.to_string()))
        .await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("LLM returned no embedding for the question");
                return Err(anyhow::anyhow!(
                    "LLM returned no embedding for the question"
                ));
            }
            r[0].iter().map(|n| *n as f32).collect()
        }
        Err(_e) => {
            log::error!("LLM returned an error: {}", _e);
            return Err(anyhow::anyhow!(
                "LLM returned no embedding for the question"
            ));
        }
    };

    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };

    let mut out = vec![];
    match search_points(&collection_name, &p).await {
        Ok(sp) => {
            for p in sp.iter() {
                let p_text = p
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("text")
                    .unwrap()
                    .as_str()
                    .unwrap();

                let issue_or_project_id = p
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("issue_or_project_id")
                    .unwrap()
                    .as_str()
                    .unwrap();

                log::info!(
                    "Received vector score={} and text={}\n",
                    p.score,
                    p_text.chars().take(50).collect::<String>()
                );
                if p.score > 0.79 {
                    out.push((issue_or_project_id.to_string(), p_text.to_string()));
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
        }
    }
    Ok(out)
}
/* pub async fn search_collection_n(
    question: &str,
    collection_name: &str,
) -> anyhow::Result<Vec<(String, String)>> {
    let mut openai = OpenAIFlows::new();
    openai.set_retry_times(3);

    let question_vector = match openai
        .create_embeddings(EmbeddingsInput::String(question.to_string()))
        .await
    {
        Ok(r) => {
            if r.len() < 1 {
                log::error!("LLM returned no embedding for the question");
                return Err(anyhow::anyhow!(
                    "LLM returned no embedding for the question"
                ));
            }
            r[0].iter().map(|n| *n as f32).collect()
        }
        Err(_e) => {
            log::error!("LLM returned an error: {}", _e);
            return Err(anyhow::anyhow!(
                "LLM returned no embedding for the question"
            ));
        }
    };

    let p = PointsSearchParams {
        vector: question_vector,
        limit: 5,
    };

    let mut out = vec![];
    match search_points(&collection_name, &p).await {
        Ok(sp) => {
            for p in sp.iter() {
                let p_text = p
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("text")
                    .unwrap()
                    .as_str()
                    .unwrap();

                let issue_or_project_id = p
                    .payload
                    .as_ref()
                    .unwrap()
                    .get("issue_or_project_id")
                    .unwrap()
                    .as_str()
                    .unwrap();

                log::debug!(
                    "Received vector score={} and text={}",
                    p.score,
                    p_text.chars().take(50).collect::<String>()
                );
                if p.score > 0.75 {
                    out.push((issue_or_project_id.to_string(), p_text.to_string()));
                }
            }
        }
        Err(e) => {
            log::error!("Vector search returns error: {}", e);
        }
    }
    Ok(out)
} */

pub async fn create_my_collection(vector_size: u64, collection_name: &str) -> anyhow::Result<()> {
    let params = CollectionCreateParams {
        vector_size: vector_size,
    };

    if let Err(_e) = create_collection(collection_name, &params).await {
        log::info!("Collection already exists");
    }

    check_vector_db(collection_name).await;
    Ok(())
}
