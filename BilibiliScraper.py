import requests
import re
from bs4 import BeautifulSoup

def extract_video_links(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    video_links = []

    # Find all video card elements
    video_cards = soup.find_all('div', class_='bili-video-card__wrap')

    # Extract video links from each video card
    for card in video_cards:
        link_element = card.find('a', class_='')
        if link_element:
            video_links.append(link_element['href'])

    return video_links

def extract_video_id(url):
    pattern = re.compile(r'\/video\/(BV[0-9a-zA-Z]+)\/')
    match = pattern.search(url)

    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None

def extract_api_url(original_url):
    modified_url = original_url.replace('://www.bilibili.com', '://ibilibili.com')

    try:
        response = requests.get(modified_url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            api_inputs = soup.find_all('input', {'class': 'form-control'})

            for api_input in api_inputs:
                if 'api.bilibili.com' in api_input.get('value', ''):
                    api_url = api_input.get('value')
                    return api_url

            print("Error: Unable to find the API URL textbox.")
        else:
            print(f"Error: Unable to fetch the page. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")
    return None

base_url = "https://www.bilibili.com"

headers = {
'user-agent': 'https://explore.whatismybrowser.com/useragents/parse/?analyse-my-user-agent=yes'
}

def main():
    for i in range (1,35):
        print ('on ' + str(i) + ' iteration')
        search_result = 'https://search.bilibili.com/all?keyword=%E4%B8%AD%E5%9B%BD%20%E9%BB%91%E4%BA%BA&from_source=webtop_search&spm_id_from=333.1007&search_source=3'
        if (i==1):
            response = requests.get(search_result, headers=headers)
        else:
            edited_result = search_result + '&page=' + str(i) + '&o=' + str((i-1)*30)
            response = requests.get(edited_result, headers=headers)
        if response.status_code == 200:
            while True:
                # Extract video links from the current page
                video_links = extract_video_links(response.text)
                with open('C:\\Users\\CharlesQX\\Desktop\\python\\ResearchProjectData\\video_list.txt', mode='r', encoding='utf-8') as file:
                    existing_links = file.read()
                with open('C:\\Users\\CharlesQX\\Desktop\\python\\ResearchProjectData\\video_list.txt', mode='a', encoding='utf-8') as file:
                    for video_link in video_links:
                        if video_link not in existing_links:
                            print ('https:' + video_link)
                            file.write('https:' + video_link)
                            file.write('\n')
                            with open('C:\\Users\\CharlesQX\\Desktop\\python\\ResearchProjectData\\video_list.txt', mode='r', encoding='utf-8') as f:
                                existing_links = f.read()        
                break
        else:
            print(f"Failed to fetch the search page. Status code: {response.status_code}")
        print('finished one iteration')

print ('started running')
with open('C:\\Users\\CharlesQX\\Desktop\\python\\ResearchProjectData\\video_list.txt', 'r') as file:
    main()
    for original_url in file:
        if original_url == '':
            break
        else:
            api_url = extract_api_url(original_url)

            response = requests.get(url=api_url, headers=headers)

            response.encoding = 'utf-8'

            content_list=re.findall('<d p=".*?">(.*?)</d>', response.text)

            file_name = extract_video_id(original_url) + '.txt'

            for content in content_list:
                folder_path = 'C:\\Users\\CharlesQX\\Desktop\\python\\ResearchProjectData\\弹幕信息'
                full_path = f'{folder_path}/{file_name}'
                with open(full_path, mode='a', encoding='utf-8') as f:
                    f.write(content)
                    f.write('\n')
                    print(content)