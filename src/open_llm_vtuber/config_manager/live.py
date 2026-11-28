"""Configuration models for live streaming platform integrations."""

from typing import ClassVar

from pydantic import Field

from .i18n import Description, I18nMixin


class BiliBiliLiveConfig(I18nMixin):
    """Configuration for BiliBili Live platform."""

    room_ids: list[int] = Field([], alias="room_ids")
    sessdata: str = Field("", alias="sessdata")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "room_ids": Description(
            en="List of BiliBili live room IDs to monitor", zh="要监控的B站直播间ID列表"
        ),
        "sessdata": Description(
            en="SESSDATA cookie value for authenticated requests (optional)",
            zh="用于认证请求的SESSDATA cookie值（可选）",
        ),
    }


class YouTubeLiveConfig(I18nMixin):
    """Configuration for YouTube Live chat polling."""

    video_ids: list[str] = Field([], alias="video_ids")
    api_key: str = Field("", alias="api_key")
    poll_interval_seconds: float = Field(5.0, alias="poll_interval_seconds")
    include_author: bool = Field(True, alias="include_author")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "video_ids": Description(
            en="YouTube live video IDs to monitor", zh="要监控的YouTube直播视频ID列表"
        ),
        "api_key": Description(
            en="YouTube Data API key; can be supplied with YOUTUBE_API_KEY",
            zh="YouTube Data API密钥；也可以使用YOUTUBE_API_KEY环境变量",
        ),
        "poll_interval_seconds": Description(
            en="Minimum live chat polling interval in seconds",
            zh="直播聊天轮询的最小间隔秒数",
        ),
        "include_author": Description(
            en="Include the chat author name in forwarded messages",
            zh="在转发的消息中包含聊天作者名称",
        ),
    }


class KickLiveConfig(I18nMixin):
    """Configuration for Kick official Events API webhook integration."""

    channel_slugs: list[str] = Field([], alias="channel_slugs")
    client_id: str = Field("", alias="client_id")
    client_secret: str = Field("", alias="client_secret")
    webhook_host: str = Field("0.0.0.0", alias="webhook_host")
    webhook_port: int = Field(8765, alias="webhook_port")
    webhook_path: str = Field("/kick/events", alias="webhook_path")
    include_author: bool = Field(True, alias="include_author")
    verify_signatures: bool = Field(True, alias="verify_signatures")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "channel_slugs": Description(
            en="Kick channel slugs to monitor", zh="要监控的Kick频道slug列表"
        ),
        "client_id": Description(
            en="Kick developer application client ID; can use KICK_CLIENT_ID",
            zh="Kick开发者应用client ID；也可以使用KICK_CLIENT_ID环境变量",
        ),
        "client_secret": Description(
            en="Kick developer application client secret; can use KICK_CLIENT_SECRET",
            zh="Kick开发者应用client secret；也可以使用KICK_CLIENT_SECRET环境变量",
        ),
        "webhook_host": Description(
            en="Local webhook listener host/interface", zh="本地webhook监听主机或接口"
        ),
        "webhook_port": Description(
            en="Local webhook listener port", zh="本地webhook监听端口"
        ),
        "webhook_path": Description(
            en="Webhook path matching the public URL configured in Kick",
            zh="与Kick中配置的公开URL匹配的webhook路径",
        ),
        "include_author": Description(
            en="Include the sender name in forwarded messages",
            zh="在转发的消息中包含发送者名称",
        ),
        "verify_signatures": Description(
            en="Verify Kick webhook signatures before processing",
            zh="处理前验证Kick webhook签名",
        ),
    }


class LiveConfig(I18nMixin):
    """Configuration for live streaming platforms integration."""

    bilibili_live: BiliBiliLiveConfig = Field(
        BiliBiliLiveConfig(), alias="bilibili_live"
    )
    youtube_live: YouTubeLiveConfig = Field(YouTubeLiveConfig(), alias="youtube_live")
    kick_live: KickLiveConfig = Field(KickLiveConfig(), alias="kick_live")

    DESCRIPTIONS: ClassVar[dict[str, Description]] = {
        "bilibili_live": Description(
            en="Configuration for BiliBili Live platform", zh="B站直播平台配置"
        ),
        "youtube_live": Description(
            en="Configuration for YouTube Live platform", zh="YouTube直播平台配置"
        ),
        "kick_live": Description(
            en="Configuration for Kick live platform", zh="Kick直播平台配置"
        ),
    }
