from pydantic_settings import BaseSettings
from pydantic import SecretStr


class BackendSettings(BaseSettings):
    app_name: str = "Backend Application"
    env_mode: str

    # database information
    database_url: SecretStr

    # llm informations
    openai_api_key: SecretStr
    gemin_api_key: SecretStr
    groq_api_key: SecretStr

    # web-search api's
    serpapi_api_key: SecretStr
    travily_api_key: SecretStr

    # redis information
    redis_url: SecretStr

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = BackendSettings()
