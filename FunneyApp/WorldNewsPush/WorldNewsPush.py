import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import schedule
import time

# 获取百度新闻最新消息
def get_baidu_news():
    url = "https://news.baidu.com/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"获取新闻失败: {e}")
        return "获取新闻失败，请稍后重试。"
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.text, 'html.parser')
    # 百度新闻首页的新闻链接通常在a标签，带有href且有新闻标题
    news_list = soup.select('a[href^="http"]')
    news_summary = []
    count = 0
    for news in news_list:
        title = news.get_text(strip=True)
        link = news.get('href')
        # 过滤无标题或无效链接
        if title and link and len(title) > 8 and "baidu" not in link:
            news_summary.append(f"{title}\n{link}\n")
            count += 1
        if count >= 10:  # 只取前10条
            break
    if not news_summary:
        return "未能获取到新闻内容。"
    return "\n".join(news_summary)

# 获取广州未来天气信息（使用和风天气免费API或中国天气网）
def get_guangzhou_weather():
    try:
        # 使用中国天气网广州天气页面
        url = "http://www.weather.com.cn/weather/101280101.shtml"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        # 获取7天天气预报
        weather_info = []
        forecast = soup.select("ul.t.clearfix li")
        for day in forecast[:7]:  # 取未来7天
            date_tag = day.find("h1")
            weather_tag = day.find("p", class_="wea")
            temp_tag = day.find("p", class_="tem")
            if not (date_tag and weather_tag and temp_tag):
                continue
            date = date_tag.get_text(strip=True)
            weather = weather_tag.get_text(strip=True)
            temp = temp_tag.get_text(strip=True)
            weather_info.append(f"{date}: {weather}, {temp}")
        if not weather_info:
            return "未能获取到广州天气信息。"
        return "广州未来7天天气预报：\n" + "\n".join(weather_info)
    except Exception as e:
        return f"获取广州天气失败: {e}"

# 发送邮件
def send_email(news_content, to_email="20201110886@stu.gzucm.edu.cn"):
    # 邮件服务器配置
    mail_host = "smtp.163.com"  # 邮件服务器地址
    mail_user = "kevin1737@163.com"  # 发件人邮箱
    mail_pass = "BJh5muwqkZUBTZCM"  # 发件人邮箱密码（需使用授权码）
    # 邮件内容
    message = MIMEText(news_content, 'plain', 'utf-8')
    message['From'] = Header("新闻推送系统", 'utf-8')
    message['To'] = Header("用户", 'utf-8')
    message['Subject'] = Header("百度新闻最新消息汇总", 'utf-8')
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)  # 163邮箱用SSL和465端口
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(mail_user, [to_email], message.as_string())
        smtpObj.quit()
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(f"邮件发送失败: {e}")

# 通过微信推送新闻（基于wxpusher）
def send_wechat(news_content, uids=None, app_token=None):
    """
    uids: list, wxpusher 用户UID列表
    app_token: str, wxpusher APP_TOKEN
    """
    if not uids or not app_token:
        print("未配置wxpusher的UID或APP_TOKEN，无法发送微信推送。")
        return
    url = "https://wxpusher.zjiecode.com/api/send/message"
    data = {
        "appToken": app_token,
        "content": news_content,
        "summary": "百度新闻最新消息汇总",  # 消息摘要
        "contentType": 1,  # 1表示文本
        "uids": uids
    }
    try:
        resp = requests.post(url, json=data, timeout=10)
        if resp.status_code == 200 and resp.json().get("code") == 1000:
            print("微信推送成功")
        else:
            print(f"微信推送失败: {resp.text}")
    except Exception as e:
        print(f"微信推送异常: {e}")

def push_news():
    news_content = get_baidu_news()
    weather_content = get_guangzhou_weather()
    full_content = f"{weather_content}\n\n{news_content}"
    send_email(full_content)
    # 配置你的wxpusher参数
    wxpusher_app_token = "AT_hXpgb6vEw5eCrFv8AtfbV4H1k5lyW3de"  # 替换为你的APP_TOKEN
    wxpusher_uids = ["UID_yoal0CBpdU95qe2nLmvUmUQBZfDV"]           # 替换为你的UID列表
    send_wechat(full_content, uids=wxpusher_uids, app_token=wxpusher_app_token)

# 主程序
if __name__ == "__main__":
    push_times = input("请输入每天推送的时间（多个时间用英文逗号分隔，如 08:00,12:00,18:30）：").strip()
    try:
        times = [t.strip() for t in push_times.split(",") if t.strip()]
        if not times:
            raise ValueError("未输入有效的时间点。")
        for t in times:
            schedule.every().day.at(t).do(push_news)
            print(f"已设置每天{t}定时推送新闻和天气。")
        print("按Ctrl+C退出。")
        while True:
            schedule.run_pending()
            time.sleep(30)
    except Exception as e:
        print(f"定时任务设置失败: {e}")