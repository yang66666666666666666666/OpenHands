"""
OpenHands AWS Bedrock 支持模块

本模块提供了对 AWS Bedrock 服务的特殊支持和集成，包括：
- 基础模型列表查询
- AWS 认证和区域配置
- Bedrock 特定的模型管理

技术栈:
- Boto3: AWS SDK for Python
- AWS Bedrock: 托管的基础模型服务
"""

import boto3

from openhands.core.logger import openhands_logger as logger


def list_foundation_models(
    aws_region_name: str, aws_access_key_id: str, aws_secret_access_key: str
) -> list[str]:
    """
    列出 AWS Bedrock 中可用的基础模型。
    
    查询指定 AWS 区域中支持文本输出和按需推理的基础模型列表。
    
    参数:
        aws_region_name: AWS 区域名称
        aws_access_key_id: AWS 访问密钥 ID
        aws_secret_access_key: AWS 秘密访问密钥
        
    返回:
        可用模型 ID 的列表
    """
    try:
        # The AWS bedrock model id is not queried, if no AWS parameters are configured.
        client = boto3.client(
            service_name='bedrock',
            region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        foundation_models_list = client.list_foundation_models(
            byOutputModality='TEXT', byInferenceType='ON_DEMAND'
        )
        model_summaries = foundation_models_list['modelSummaries']
        return ['bedrock/' + model['modelId'] for model in model_summaries]
    except Exception as err:
        logger.warning(
            '%s. Please config AWS_REGION_NAME AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY'
            ' if you want use bedrock model.',
            err,
        )
        return []


def remove_error_modelId(model_list: list[str]) -> list[str]:
    return list(filter(lambda m: not m.startswith('bedrock'), model_list))
