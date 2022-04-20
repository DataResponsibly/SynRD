import sys
import pandas as pd
import concurrent.futures

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


def setup():
    sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')


def selenium_page_collection(url):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    browser = webdriver.Chrome('chromedriver', options=chrome_options)
    #browser.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS);
    
    browser.get(url)
    return browser


def selenium_data_retrieval(start_val, url, element):
    EMPTY_STRING = ""
    rows = element.find_elements(By.CSS_SELECTOR, "div.searchResult.row[data-reactid^=\.0\.0\.1\.7\.0\.2]")

    outputs = []
    for row_idx, row in enumerate(rows):
        result = {}
        try:
            result["start_val"] = start_val
            result["url"] = url
            field_attempting = "year"
            result["year"] = row.find_elements(By.CSS_SELECTOR, "[title^=Publication]")[0].text
            
            field_attempting = "displayCitation"
            result["displayCitation"] = row.find_elements(By.CLASS_NAME, "displayCitation")[0].text

            field_attempting = "fullTexts"
            if row.find_elements(By.CLASS_NAME, "fullText"):
                fullTexts = [link_el.get_attribute("href") for link_el in 
                            row.find_elements(By.CLASS_NAME, "fullText")[0].find_elements(By.CSS_SELECTOR, "a[href]")]
            else:
                fullTexts = []
            
            for i in range(20):
                if i < len(fullTexts):
                    result["fullText"+str(i)] = fullTexts[i]
                else:
                    result["fullText"+str(i)] = EMPTY_STRING
            
            field_attempting = "relevantStudies"
            relStudies = [link_el.get_attribute("href") for link_el in 
                         row.find_elements(By.CLASS_NAME, "relStudies")[0].find_elements(By.CSS_SELECTOR, "a[href]")]
            for i in range(100):
                if i < len(relStudies):
                    result["relStudies"+str(i)] = relStudies[i]
                else:
                    result["relStudies"+str(i)] = EMPTY_STRING
                
            outputs.append(result)
        except:
            print(f"Suppressing exception due to {field_attempting} for row {row_idx} of url {url}.")

    return outputs


def threaded_selenium_tasks(start_val, url):
    return selenium_data_retrieval(start_val, url, selenium_page_collection(url))


def test_single_icpsr(rows=5):
    test_url = 'https://www.icpsr.umich.edu/web/ICPSR/search/publications?start=0&COLLECTION=DATA&ARCHIVE=ICPSR&sort=YEAR_PUB_DATE%20desc%2CAUTHORS_SORT%20asc&rows='+str(rows)
    return threaded_selenium_tasks(0, test_url)


def test_threaded_icpsr(num_rows, num_pages):
    return run_icpsr(num_results=num_rows, 
                     max_publications=num_rows*num_pages)


def run_icpsr(num_results=500, max_publications=101926, start=0):
    links = []
    
    for start_val in range(start, max_publications, num_results):
        links.append((start_val, start_val + num_results,
                     f"https://www.icpsr.umich.edu/web/ICPSR/search/publications?" +
                     f"start={start_val}&COLLECTION=DATA&ARCHIVE=ICPSR&sort=YEAR_" +
                     f"PUB_DATE%20desc%2CAUTHORS_SORT%20asc&rows={num_results}"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(threaded_selenium_tasks, start_val, url) for start_val, _, url in links
        }

        for fut in concurrent.futures.as_completed(futures):
            print(f"Future starting at {fut.result()[0]['start_val']} completed with {len(fut.result())} rows.")
            pd.DataFrame(fut.result()).to_csv(f"result_{fut.result()[0]['start_val']}.csv", index=False)
    
    return
