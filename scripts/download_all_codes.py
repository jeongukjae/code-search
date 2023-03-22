import os
import requests

from absl import logging, flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("org", None, "Github organization name.", required=True)
flags.DEFINE_bool("user", False, "Whether the org is a user.")
flags.DEFINE_string(
    "output_dir", None, "Output directory to save the codes.", required=True
)
flags.DEFINE_string("github_token_env_key", "GITHUB_TOKEN", "Github token env key.")
flags.DEFINE_bool("exclude_forks", True, "Whether to exclude forked repositories.")

# details
flags.DEFINE_integer("list_repo_per_page", 100, "Number of repos to list per page.")


def main(argv):
    if not FLAGS.output_dir:
        raise ValueError("Output directory is not specified.")

    github_token = os.environ.get(FLAGS.github_token_env_key)
    if not github_token:
        raise ValueError(
            f"Github token is not found in env key {FLAGS.github_token_env_key}"
        )

    page = 1
    repositories = []
    while True:
        logging.info(f"Fetching repositories page {page}...")
        resource_type = "users" if FLAGS.user else "orgs"
        url = f"https://api.github.com/{resource_type}/{FLAGS.org}/repos"
        response = requests.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            params={
                "per_page": FLAGS.list_repo_per_page,
                "page": page,
            },
        )
        if response.status_code != 200:
            response.raise_for_status()

        repository_list = response.json()
        if len(repository_list) == 0:
            break

        if FLAGS.exclude_forks:
            repository_list = [r for r in repository_list if not r["fork"]]
        repositories.extend(repository_list)
        page += 1

    logging.info(f"Found {len(repositories)} repositories.")
    for repository in repositories:
        repo_name = repository["full_name"]
        logging.info(f"Downloading {repo_name}...")

        branch = repository["default_branch"]
        url = f"https://api.github.com/repos/{repo_name}/branches/{branch}"
        branch_response = requests.get(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        if branch_response.status_code != 200:
            logging.info(f"Failed to get branch {branch} for {repo_name}. Skipping...")
            continue

        ref = branch_response.json()["commit"]["sha"]
        url = f"https://api.github.com/repos/{repo_name}/zipball/{ref}"
        with requests.get(
            url,
            stream=True,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {github_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        ) as r:
            r.raise_for_status()
            repo_name_without_owner = repo_name.split("/")[1]
            output = os.path.join(
                FLAGS.output_dir, f"{repo_name_without_owner}_{ref}.zip"
            )
            with open(output, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)


if __name__ == "__main__":
    app.run(main)
