use crate::issue_tracker::*;
use crate::llm_utils::parse_summary_and_keywords;
use crate::llm_utils_together::*;
use dotenv::dotenv;
use mysql_async::prelude::*;
use mysql_async::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct IssueOut {
    pub issue_id: String,
    pub project_id: String,
    pub project_logo: String,
    pub main_language: String,
    pub repo_stars: i32,
    pub issue_title: String,
    pub issue_creator: String,
    pub issue_description: String,
    pub issue_budget: Option<i32>,
    pub issue_assignees: Option<String>,
    pub issue_linked_pr: Option<String>,
    pub issue_status: Option<String>,
    pub review_status: String,
    #[serde(default = "default_value")]
    pub issue_budget_approved: bool,
    pub running_budget: (i32, i32, i32),
    pub issue_stats: (i32, i32, i32, i32),
}

fn default_value() -> bool {
    false
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub enum ReviewStatus {
    #[default]
    Queue,
    Approve,
    Decline,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ProjectOut {
    pub project_id: String,
    pub project_logo: Option<String>,
    pub main_language: Option<String>,
    pub repo_stars: i32,
    pub project_description: Option<String>,
    pub issues_list: Option<Vec<String>>,
    pub total_budget_allocated: Option<i32>,
    pub total_count: i32,
}

pub async fn get_pool() -> Pool {
    dotenv().ok();
    let url = std::env::var("DATABASE_URL").expect("not url db url found");

    let opts = Opts::from_url(&url).unwrap();
    let builder = OptsBuilder::from_opts(opts);
    // The connection pool will have a min of 5 and max of 10 connections.
    let constraints = PoolConstraints::new(5, 10).unwrap();
    let pool_opts = PoolOpts::default().with_constraints(constraints);

    Pool::new(builder.pool_opts(pool_opts))
}

pub async fn project_exists(pool: &mysql_async::Pool, project_id: &str) -> anyhow::Result<bool> {
    let mut conn = pool.get_conn().await?;
    let result: Option<u32> = conn
        .query_first(format!(
            "SELECT 1 FROM projects WHERE project_id = '{}'",
            project_id
        ))
        .await?;

    match result {
        Some(_) => Ok(true),
        None => {
            log::error!("Project not found");
            Err(anyhow::Error::msg("Project not found"))
        }
    }
}

pub async fn fill_project_w_repo_data(pool: &Pool, repo_data: RepoData) -> anyhow::Result<()> {
    let mut conn = pool.get_conn().await?;

    let project_description = if !repo_data.repo_description.is_empty() {
        repo_data.repo_description.clone()
    } else if !repo_data.repo_readme.is_empty() {
        repo_data.repo_readme.chars().take(1000).collect::<String>()
    } else {
        String::from("No description available")
    };

    if   let Err(e) = conn
        .exec_drop(
            r"INSERT INTO projects (project_id, project_logo, main_language, repo_stars, project_description)
        VALUES (:project_id, :project_logo, :main_language, :repo_stars, :project_description)
        ON DUPLICATE KEY UPDATE
        project_logo = VALUES(project_logo),
        main_language = VALUES(main_language),
        repo_stars = VALUES(repo_stars),
        project_description = VALUES(project_description);",
            params! {
                "project_id" => &repo_data.project_id,
                "project_logo" => &repo_data.project_logo,
                "main_language" => &repo_data.main_language,
                "repo_stars" => repo_data.repo_stars,
                "project_description" => project_description,
            },
        )
        .await

    {
            log::error!("Failed to fill project with repo data: {:?}", e);
    }

    Ok(())
}

pub async fn issue_exists(pool: &mysql_async::Pool, issue_id: &str) -> anyhow::Result<bool> {
    let mut conn = pool.get_conn().await?;
    let result: Option<u32> = conn
        .query_first(format!(
            "SELECT 1 FROM issues WHERE issue_id = '{}'",
            issue_id
        ))
        .await?;

    match result {
        Some(_) => Ok(true),
        None => {
            log::error!("Issue not found");
            Err(anyhow::Error::msg("Issue not found"))
        }
    }
}

pub async fn pull_request_exists(pool: &mysql_async::Pool, pull_id: &str) -> anyhow::Result<bool> {
    let mut conn = pool.get_conn().await?;
    let result: Option<u32> = conn
        .query_first(format!(
            "SELECT 1 FROM pull_requests WHERE pull_id = '{}'",
            pull_id
        ))
        .await?;

    match result {
        Some(_) => Ok(true),
        None => {
            log::error!("Pull request not found");
            Err(anyhow::Error::msg("Pull request not found"))
        }
    }
}

pub async fn add_issues_open(pool: &Pool, issue: &IssueOpen) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let query = r"INSERT INTO issues_open (issue_id, project_id, issue_title, issue_creator, issue_budget, issue_description)
                  VALUES (:issue_id, :project_id, :issue_title, :issue_creator, :issue_budget, :issue_description)";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_id" => &issue.issue_id,
                "project_id" => &issue.project_id,
                "issue_title" => &issue.issue_title,
                "issue_creator" => &issue.issue_creator,
                "issue_budget" => &issue.issue_budget,
                "issue_description" => &issue.issue_description,
            },
        )
        .await
    {
        // log::error!("Error add issues_open: {:?}", e);
    };

    Ok(())
}

pub async fn add_issues_comment(pool: &Pool, issue: IssueComment) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let query = r"INSERT INTO issues_comment (issue_id, comment_creator, comment_date, comment_body)
    SELECT :issue_id, :comment_creator, :comment_date, :comment_body
    FROM dual
    WHERE NOT EXISTS (
        SELECT 1 FROM issues_comment
        WHERE issue_id = :issue_id AND comment_date = :comment_date
    ) LIMIT 1;";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_id" => &issue.issue_id,
                "comment_creator" => &issue.comment_creator,
                "comment_date" => &issue.comment_date,
                "comment_body" => &issue.comment_body,
            },
        )
        .await
    {
        if let mysql_async::Error::Server(server_error) = &e {
            if server_error.code == 23000 {
                log::info!("Skipping duplicate comment: {:?}", issue);
                return Ok(());
            }
        }
        log::error!("Error add issues_comment: {:?}", e);
        return Err(e.into());
    }

    Ok(())
}
pub async fn add_issues_open_batch(pool: &Pool, issues: Vec<IssueOpen>) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let query = r"INSERT INTO issues_open (issue_id, project_id, issue_title, issue_budget, issue_description)
                  VALUES (:issue_id, :project_id, :issue_title, :issue_budget, :issue_description)";

    if let Err(e) = query
        .with(issues.iter().map(|issue| {
            params! {
                "issue_id" => &issue.issue_id,
                "project_id" => &issue.project_id,
                "issue_title" => &issue.issue_title,
                "issue_budget" => &issue.issue_budget,
                "issue_description" => &issue.issue_description,
            }
        }))
        .batch(&mut conn)
        .await
    {
        log::error!("Error add issues_open in batch: {:?}", e);
    };

    Ok(())
}

pub async fn add_issues_closed(pool: &Pool, issue: IssueClosed) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let issue_assignees_json: Value = json!(issue.issue_assignees).into();

    let query = r"INSERT INTO issues_closed (issue_id, issue_assignees, issue_linked_pr)
                  VALUES (:issue_id, :issue_assignees, :issue_linked_pr)";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_id" => &issue.issue_id,
                "issue_assignees" => &issue_assignees_json,
                "issue_linked_pr" => issue.issue_linked_pr.as_deref(),
            },
        )
        .await
    {
        log::error!("Error add issues_closed: {:?}", e);
    };

    Ok(())
}

pub async fn add_issues_assigned(pool: &Pool, issue_assigned: IssueAssigned) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let issue_assignee = if issue_assigned.issue_assignee.is_empty() {
        None
    } else {
        Some(issue_assigned.issue_assignee)
    };

    let query = r"INSERT INTO issues_assigned (issue_id, issue_assignee, date_assigned)
                  VALUES (:issue_id, :issue_assignee, :date_assigned)";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_id" => &issue_assigned.issue_id,
                "issue_assignee" => &issue_assignee,
                "date_assigned" => &issue_assigned.date_assigned,
            },
        )
        .await
    {
        log::error!("Error add issues_assigned: {:?}", e);
    };

    Ok(())
}

pub async fn mark_id_indexed(pool: &Pool, issue_or_project_id: &str) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let query = r"UPDATE issues_repos_summarized
    SET indexed=1 WHERE issue_or_project_id = :issue_or_project_id";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_or_project_id" => &issue_or_project_id,
            },
        )
        .await
    {
        log::error!("Error marking issue_or_project_id: {:?}", e);
    };

    Ok(())
}

pub async fn add_or_update_summary_and_id(
    pool: &Pool,
    issue_or_project_id: &str,
    issue_or_project_summary: &str,
    keyword_tags: Vec<String>,
) -> Result<()> {
    let mut conn = pool.get_conn().await?;
    let keyword_tags_json_str = json!(keyword_tags).to_string();

    let query = r"INSERT INTO issues_repos_summarized (issue_or_project_id, issue_or_project_summary, keyword_tags)
    VALUES (:issue_or_project_id, :issue_or_project_summary, :keyword_tags_json_str)
    ON DUPLICATE KEY UPDATE
    keyword_tags = :keyword_tags_json_str;";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "issue_or_project_id" => &issue_or_project_id,
                "issue_or_project_summary" => &issue_or_project_summary,
                "keyword_tags_json_str" => &keyword_tags_json_str,
            },
        )
        .await
    {
        // Log the error if the query fails
        log::error!("Error adding or updating issue_or_project_id: {:?}", e);
        return Err(e);
    };

    Ok(())
}


pub async fn add_pull_request(pool: &Pool, pull: OuterPull) -> Result<()> {
    let mut conn = pool.get_conn().await?;

    let query = r"INSERT INTO pull_requests (pull_id, pull_title, pull_author, project_id, date_merged)
                  VALUES (:pull_id, :pull_title, :pull_author, :project_id, :date_merged)";

    if let Err(e) = conn
        .exec_drop(
            query,
            params! {
                "pull_id" => &pull.pull_id,
                "pull_title" => &pull.pull_title,
                "pull_author" => pull.pull_author.as_deref(),
                "project_id" => &pull.project_id,
                "date_merged" => pull.merged_at,
            },
        )
        .await
    {
        log::error!("Error add pull_request: {:?}", e);
    };

    Ok(())
}

pub async fn get_issues_repos_from_db() -> Result<Vec<(String, String)>> {
    let pool = get_pool().await;
    let mut conn = pool.get_conn().await?;

    let query = r"SELECT issue_or_project_id, issue_or_project_summary FROM issues_repos_summarized WHERE indexed=0 limit 50";

    let entries: Vec<(String, String)> = conn
        .query_map(
            query,
            |(issue_or_project_id, issue_or_project_summary): (String, String)| {
                (issue_or_project_id, issue_or_project_summary)
            },
        )
        .await?;

    Ok(entries)
}

pub async fn get_issues_from_db() -> Result<Vec<(String, String, String, Option<String>)>> {
    let pool = get_pool().await;
    let mut conn = pool.get_conn().await?;

    let query = r"SELECT issue_id, issue_title, issue_description, issue_assignees FROM issues_master WHERE issue_id not in (SELECT issue_or_project_id FROM issues_repos_summarized) limit 50";

    let issues: Vec<(String, String, String, Option<String>)> = conn
        .query_map(
            query,
            |(issue_id, issue_title, issue_description, issue_assignees): (
                String,
                String,
                String,
                Option<String>,
            )| { (issue_id, issue_title, issue_description, issue_assignees) },
        )
        .await?;

    Ok(issues)
}

pub async fn summarize_issue_add_in_db(pool: &Pool, issue: &IssueOpen) -> anyhow::Result<()> {
    let issue_clone = issue.clone();
    let issue_title = issue_clone.issue_title;
    let issue_id = issue_clone.issue_id;
    let issue_description = issue_clone.issue_description;
    log::info!("Summarizing issue: {}",issue_id);

    let parts: Vec<&str> = issue_id.split('/').collect();
    let owner = parts[3].to_string();
    let repo = parts[4].to_string();

    let system_prompt_long_input = r#"
        Summarize the GitHub issue in one paragraph without mentioning the issue number. Highlight the key problem and any signature information provided. The summary should be concise, informative, and easy to understand, prioritizing clarity and brevity. Additionally, extract high-level keywords that represent broader categories or themes relevant to the issue's purpose, features, and tools used. These keywords should help categorize the issue in a wider context and should not be too literal or specific, avoiding overly long phrases unless absolutely necessary. Expected Output:
        { \"summary\": \"the_summary_generated, a short paragraph summarizing the issue, including its purpose and features, without referencing the issue number.\",
          \"keywords\": [\"a list of high-level keywords that encapsulate the broader context, categories, or themes of the issue, excluding specific details and issue numbers.\"] }
        Ensure you reply in RFC8259-compliant JSON format."#;

        let system_prompt_short_input = r#"
        Given the limited information available, summarize the GitHub issue in one paragraph without mentioning the issue number. Highlight the key problem and any signature information that can be inferred. The summary should be concise, informative, and easy to understand, prioritizing clarity and brevity even with scant details. Additionally, extract high-level keywords that represent broader categories or themes relevant to the issue's inferred purpose, features, and tools used. These keywords should help categorize the issue in a wider context and should not be too literal or specific, avoiding overly long phrases unless absolutely necessary. Expected Output:
        { \"summary\": \"The summary generated should be a concise paragraph that highlights any discernible purpose, technologies, or features from the limited information.\",
          \"keywords\": [\"A list of inferred high-level keywords that broadly categorize the repository based on the scant details available.\"] }
        Ensure you reply in RFC8259-compliant JSON format."#;

    let generated_summary = if issue_description.len() < 200 {
        let raw_input_texts = format!(
                "Here is the input: `{issue_title}` at repository `{repo}` by owner `{owner}`, states: {issue_description}"
            );
        chat_inner_async(system_prompt_short_input, &raw_input_texts, 180).await?
    } else {
        let raw_input_texts=  format!(
                "Here is the input: The issue titled `{issue_title}` at repository `{repo}` by owner `{owner}`, states in the body text: {issue_description}"
            ).chars().take(4000).collect::<String>();
        chat_inner_async(system_prompt_long_input, &raw_input_texts, 250).await?
    };

    let (summary, keyword_tags) = parse_summary_and_keywords(&generated_summary);
    // log::info!("{}, {:?}", issue_id, keyword_tags.clone());
    let _ = add_or_update_summary_and_id(&pool, &issue_id, &summary, keyword_tags).await;

    Ok(())
}

pub async fn summarize_project_add_in_db(pool: &Pool, repo_data: RepoData) -> anyhow::Result<()> {
    let parts: Vec<&str> = repo_data.project_id.split('/').collect();
    let owner = parts[3].to_string();
    let repo = parts[4].to_string();
     log::info!("Summarizing repo: {}",repo_data.project_id);

    let project_descrpition = repo_data.repo_description;
    let project_readme = repo_data.repo_readme;
    let main_language = repo_data.main_language;

    let use_lang_str = if main_language.is_empty() {
        String::from("")
    } else {
        format!("mainly uses `{main_language}` in the project")
    };

    let project_readme_str = match project_readme.is_empty() {
        false => format!("states in readme: {project_readme}"),
        true => String::from(""),
    };

    let system_prompt_long_input = r#"
    Summarize the GitHub repository's README or description in one detailed paragraph, focusing solely on the essential aspects such as the project's purpose, technologies used, and notable features. Do not include non-essential elements like personal appeals or donation links. Extract high-level keywords that represent broader categories or themes relevant to the project. These keywords should categorize the project in a wider context and not be overly specific or literal. Expected Output:
    { \"summary\": \"A comprehensive paragraph that succinctly summarizes the repository, highlighting its purpose, technologies, and key features, without including extraneous details.\",
      \"keywords\": [\"A list of high-level keywords that encapsulate the broader context, categories, or themes of the repository, focusing on essential aspects only.\"] }
    Ensure your reply is in RFC8259-compliant JSON format.
    "#;

    let system_prompt_short_input = r#"When summarizing a GitHub repository's README or description, concentrate on the core content. Provide a concise paragraph that captures the primary purpose, technologies used, and notable features. Avoid mentioning non-essential elements such as donation links or personal appeals. Deduce and include high-level keywords that broadly categorize the repository, focusing on the technologies, functionality, and scope based on the available information. These keywords should reflect the main themes or categories relevant to the project. Expected Output:
    { \"summary\": \"The summary generated should be a concise paragraph that highlights any discernible purpose, technologies, or features from the limited information.\",
      \"keywords\": [\"A list of inferred high-level keywords that broadly categorize the repository based on the scant details available.\"] }
    Ensure you reply in RFC8259-compliant JSON format."#;

    let generated_summary = if project_readme.len() < 200 {
        let raw_input_texts = format!(
            "Here is the input: The repository `{repo}` by owner `{owner}` {use_lang_str},`{project_descrpition}`, {project_readme_str}"
        );

        chat_inner_async(system_prompt_short_input, &raw_input_texts, 180).await?
    } else {
        let raw_input_texts = format!(
                "Here is the input: The repository `{repo}`  by owner `{owner}` {use_lang_str}, has a short text description: `{project_descrpition}`, mentioned more details in readme: `{project_readme}`"
            ).chars().take(4000).collect::<String>();

        chat_inner_async(system_prompt_long_input, &raw_input_texts, 250).await?
    };
    //  log::info!("generated summary: {}", generated_summary.to_string());

    let (summary, keyword_tags) = parse_summary_and_keywords(&generated_summary);
    //  log::info!("keywords: {:?}", &keyword_tags);

    let _ =
        add_or_update_summary_and_id(&pool, &repo_data.project_id, &summary, keyword_tags).await;
    Ok(())
}
