import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import schedule
import time
import os

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
    news_list = soup.select('a[href^="http"]')
    news_summary = []
    count = 0
    for news in news_list:
        title = news.get_text(strip=True)
        link = news.get('href')
        if title and link and len(title) > 8 and "baidu" not in link:
            news_summary.append(f"{title}\n{link}\n")
            count += 1
        if count >= 10:
            break
    if not news_summary:
        return "未能获取到新闻内容。"
    return "\n".join(news_summary)

# 获取广州未来天气信息（含天气图片）
def get_guangzhou_weather():
    try:
        url = "http://www.weather.com.cn/weather/101280101.shtml"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = resp.apparent_encoding
        soup = BeautifulSoup(resp.text, "html.parser")
        weather_info = []
        weather_images = []
        forecast = soup.select("ul.t.clearfix li")
        for idx, day in enumerate(forecast[:7]):
            date_tag = day.find("h1")
            weather_tag = day.find("p", class_="wea")
            temp_tag = day.find("p", class_="tem")
            img_tag = day.find("big", class_="png") or day.find("img")
            if not (date_tag and weather_tag and temp_tag):
                continue
            date = date_tag.get_text(strip=True)
            weather = weather_tag.get_text(strip=True)
            temp = temp_tag.get_text(strip=True)
            img_url = None
            if img_tag:
                if img_tag.has_attr("src"):
                    img_url = img_tag["src"]
                elif img_tag.has_attr("style"):
                    style = img_tag["style"]
                    if "url(" in style:
                        img_url = style.split("url(")[1].split(")")[0].strip("'\"")
            if img_url and not img_url.startswith("http"):
                img_url = "http:" + img_url
            weather_info.append({
                "date": date,
                "weather": weather,
                "temp": temp,
                "img_url": img_url,
                "img_id": f"weatherimg{idx}"
            })
            if img_url:
                weather_images.append({"url": img_url, "cid": f"weatherimg{idx}"})
        if not weather_info:
            return "未能获取到广州天气信息。", []
        html = "<h3>广州未来7天天气预报：</h3><table border='1' cellpadding='5'><tr><th>日期</th><th>天气</th><th>温度</th><th>图标</th></tr>"
        for info in weather_info:
            img_html = f"<img src='cid:{info['img_id']}' width='40'>" if info["img_url"] else ""
            html += f"<tr><td>{info['date']}</td><td>{info['weather']}</td><td>{info['temp']}</td><td>{img_html}</td></tr>"
        html += "</table>"
        return html, weather_images
    except Exception as e:
        return f"获取广州天气失败: {e}", []

# 发送邮件（支持HTML和图片）
def send_email(news_content, to_email="20201110886@stu.gzucm.edu.cn", weather_images=None):
    mail_host = "smtp.163.com"
    mail_user = "kevin1737@163.com"
    mail_pass = "BJh5muwqkZUBTZCM"
    if weather_images is None:
        weather_images = []
    msg = MIMEMultipart()
    msg['From'] = Header("新闻推送系统", 'utf-8')
    msg['To'] = Header("用户", 'utf-8')
    msg['Subject'] = Header("百度新闻最新消息汇总", 'utf-8')
    msg.attach(MIMEText(news_content, 'html', 'utf-8'))
    for img in weather_images:
        try:
            img_resp = requests.get(img["url"], timeout=10)
            img_resp.raise_for_status()
            image = MIMEImage(img_resp.content)
            image.add_header('Content-ID', f"<{img['cid']}>")
            msg.attach(image)
        except Exception as e:
            print(f"天气图片下载失败: {img['url']} 错误: {e}")
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(mail_user, [to_email], msg.as_string())
        smtpObj.quit()
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print(f"邮件发送失败: {e}")

# 通过微信推送新闻（基于wxpusher）
def send_wechat(news_content, uids=None, app_token=None):
    if not uids or not app_token:
        print("未配置wxpusher的UID或APP_TOKEN，无法发送微信推送。")
        return
    url = "https://wxpusher.zjiecode.com/api/send/message"
    data = {
        "appToken": app_token,
        "content": news_content,
        "summary": "百度新闻最新消息汇总",
        "contentType": 1,
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
    weather_content, weather_images = get_guangzhou_weather()
    news_content = get_baidu_news()
    full_content = f"{weather_content}<br><pre>{news_content}</pre>"
    send_email(full_content, weather_images=weather_images)
    wxpusher_app_token = "AT_hXpgb6vEw5eCrFv8AtfbV4H1k5lyW3de"
    wxpusher_uids = ["UID_yoal0CBpdU95qe2nLmvUmUQBZfDV"]
    send_wechat(f"广州未来7天天气预报：\n{BeautifulSoup(weather_content, 'html.parser').get_text()}\n\n{news_content}", uids=wxpusher_uids, app_token=wxpusher_app_token)

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