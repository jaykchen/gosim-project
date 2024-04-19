use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE, USER_AGENT};
use secrecy::Secret;
use serde::Deserialize;
use std::collections::HashMap;

use async_openai::{
    config::Config,
    types::{
        // ChatCompletionFunctionsArgs, ChatCompletionRequestMessage,
        ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs,
        // ChatCompletionTool, ChatCompletionToolArgs, ChatCompletionToolType,
        CreateChatCompletionRequestArgs,
    },
    Client as OpenAIClient,
};


pub async fn chain_of_chat(
    sys_prompt_1: &str,
    usr_prompt_1: &str,
    _chat_id: &str,
    gen_len_1: u16,
    usr_prompt_2: &str,
    gen_len_2: u16,
    error_tag: &str,
) -> anyhow::Result<String> {
    let mut headers = HeaderMap::new();
    let api_key = std::env::var("DEEP_API_KEY").expect("DEEP_API_KEY must be set");
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    let config = LocalServiceProviderConfig {
        // api_base: String::from("http://52.37.228.1:8080/v1"),
        api_base: String::from("https://api.deepinfra.com/v1/openai/chat/completions"),
        headers: headers,
        api_key: Secret::new(api_key),
        query: HashMap::new(),
    };

    let model = "DEEP_API_KEY-must-be-set";
    let client = OpenAIClient::with_config(config);

    let mut messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(sys_prompt_1)
            .build()
            .expect("Failed to build system message")
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(usr_prompt_1)
            .build()?
            .into(),
    ];
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(gen_len_1)
        .model(model)
        .messages(messages.clone())
        .build()?;

    // dbg!("{:?}", request.clone());

    let chat = client.chat().create(request).await?;

    match chat.choices[0].message.clone().content {
        Some(res) => {
            println!("{:?}", res);
        }
        None => {
            return Err(anyhow::anyhow!(error_tag.to_string()));
        }
    }

    messages.push(
        ChatCompletionRequestUserMessageArgs::default()
            .content(usr_prompt_2)
            .build()?
            .into(),
    );

    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(gen_len_2)
        .model(model)
        .messages(messages)
        .build()?;

    let chat = client.chat().create(request).await?;

    match chat.choices[0].message.clone().content {
        Some(res) => {
            println!("{:?}", res);
            Ok(res)
        }
        None => {
            return Err(anyhow::anyhow!(error_tag.to_string()));
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalServiceProviderConfig {
    pub api_base: String,
    pub headers: HeaderMap,
    pub api_key: Secret<String>,
    pub query: HashMap<String, String>,
}

impl Config for LocalServiceProviderConfig {
    fn headers(&self) -> HeaderMap {
        self.headers.clone()
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.api_base, path)
    }

    fn query(&self) -> Vec<(&str, &str)> {
        self.query
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect()
    }

    fn api_base(&self) -> &str {
        &self.api_base
    }

    fn api_key(&self) -> &Secret<String> {
        &self.api_key
    }
}

pub async fn chat_inner_async(
    system_prompt: &str,
    user_input: &str,
    max_token: u16,
    model: &str,
) -> anyhow::Result<String> {
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(USER_AGENT, HeaderValue::from_static("MyClient/1.0.0"));
    let config = LocalServiceProviderConfig {
        // api_base: String::from("http://10.0.0.174:8080/v1"),
        api_base: String::from("https://api.deepinfra.com/v1/openai"),
        headers: headers,
        api_key: Secret::new("lY2h5Vd5wgdyICzjOyDmmmToeU3KyLgv".to_string()),
        query: HashMap::new(),
    };

    let client = OpenAIClient::with_config(config);
    let messages = vec![
        ChatCompletionRequestSystemMessageArgs::default()
            .content(system_prompt)
            .build()
            .expect("Failed to build system message")
            .into(),
        ChatCompletionRequestUserMessageArgs::default()
            .content(user_input)
            .build()?
            .into(),
    ];
    let request = CreateChatCompletionRequestArgs::default()
        .max_tokens(max_token)
        .model(model)
        .messages(messages)
        .build()?;

    let chat = match client.chat().create(request).await {
        Ok(chat) => chat,
        Err(_e) => {
            println!("Error getting response from OpenAI: {:?}", _e);
            return Err(anyhow::anyhow!("Failed to get reply from OpenAI: {:?}", _e));
        }
    };

    match chat.choices[0].message.clone().content {
        Some(res) => {
            // println!("{:?}", chat.choices[0].message.clone());
            Ok(res)
        }
        None => Err(anyhow::anyhow!("Failed to get reply from OpenAI")),
    }
}

pub fn parse_summary_from_raw_json(input: &str) -> String {
    #[derive(Deserialize, Debug)]
    struct SummaryStruct {
        impactful: Option<String>,
        alignment: Option<String>,
        patterns: Option<String>,
        synergy: Option<String>,
        significance: Option<String>,
    }

    let summary: SummaryStruct = serde_json::from_str(input).expect("Failed to parse summary JSON");

    let mut output = String::new();

    let fields = [
        &summary.impactful,
        &summary.alignment,
        &summary.patterns,
        &summary.synergy,
        &summary.significance,
    ];

    fields
        .iter()
        .filter_map(|&field| field.as_ref()) // Convert Option<&String> to Option<&str>
        .filter(|field| !field.is_empty()) // Filter out empty strings
        .fold(String::new(), |mut acc, field| {
            if !acc.is_empty() {
                acc.push_str(" ");
            }
            acc.push_str(field);
            acc
        })
}



pub fn parse_issue_summary_from_json(input: &str) -> anyhow::Result<Vec<(String, String)>> {
    let parsed: serde_json::Map<String, serde_json::Value> = serde_json::from_str(input)?;

    let summaries = parsed
        .iter()
        .filter_map(|(key, value)| {
            if let Some(summary_str) = value.as_str() {
                Some((key.clone(), summary_str.to_owned()))
            } else {
                None
            }
        })
        .collect::<Vec<(String, String)>>(); // Collect into a Vec of tuples

    Ok(summaries)
}

pub async fn test_gen() -> anyhow::Result<()> {
    let text = include_str!("/Users/jichen/Projects/local-llm-tester/src/raw_commit.txt");
    let raw_len = text.len();
    let stripped_texts = text.chars().take(24_000).collect::<String>();
    // let stripped_texts = String::from_utf8(response).ok()?.chars().take(24_000).collect::<String>();
    let user_name = "Akihiro Suda".to_string();

    let tag_line = "commit".to_string();

    let sys_prompt_1 = format!(
        "You're a GitHub data analysis bot. Analyze the following commit patch for its content and implications."
    );

    let usr_prompt_1 = format!(
        "Examine the commit patch: {stripped_texts}, and identify the key changes and their potential impact on the project. Exclude technical specifics, code excerpts, file changes, and metadata. Highlight only those changes that substantively alter code or functionality. Provide a fact-based representation that differentiates major from minor contributions. The commit is described as: '{tag_line}'. Summarize the main changes, focusing on modifications that directly affect core functionality, and provide insight into the value or improvement the commit brings to the project."
    );

    let usr_prompt_2 = format!(
        "Based on the analysis provided, craft a concise, non-technical summary of the key technical contributions made by {user_name} this week. The summary should be a full sentence or a short paragraph suitable for a general audience, emphasizing the contributions' significance to the project."
    );

    let summary = chain_of_chat(
        &sys_prompt_1,
        &usr_prompt_1,
        "commit-99",
        512,
        &usr_prompt_2,
        128,
        "chained-prompt-commit",
    )
    .await?;

    println!("len: {} Summary: {:?}", raw_len, summary.clone());

    use std::fs::File;
    use std::io::Write;
    let mut file = File::create("commits.txt").expect("create failed");

    // Iterate over the map and write each entry to the file
    // `value.0` is the URL and `value.1` is the summary in this context
    file.write_all(summary.as_bytes()).expect("write failed");
    Ok(())
}
