## FINA4350 Midterm Project - Christy Wong Ka Hei 

### Introduction
A comparable company analysis (CCA) is used by financial analysts to evaluate the value of a company by comparing its metrics against that of other businesses in the same industry of similar size.<sup>([source](https://www.investopedia.com/terms/c/comparable-company-analysis-cca.asp ))</sup>. While it might be easier to find similar companies for large, well-known companies (ie. Google, Microsoft), finding similar companies for smaller, less well-known companies could prove to be challenging. With an unsupervised learning method like K-means clustering, it is possible to process large amounts of company descriptions and identify groups of similar companies for the purpose of CCA. In this project, I will try to cluster similar listed companies based on their company description in their Item 1 (Business) section in their Form 10-K filing.

### Data Scraping _(datascraper.py)_
I used the sec_api to scrape for Form-10Ks and extracted the text for section 1 in datascraper.py with two main steps:
1. discover urls for Form-10Ks with QueryAPI (with a query for the most recent filings)
2. feed the urls into ExtractorAPI to obtain the text

One of the challenges I faced was that the query would return urls for companies without tickers ([here's an example](https://www.sec.gov/Archives/edgar/data/1617957/000188852422003194/gsm14g24_10k-2021.htm)! ), but I found that those 10-K reports were mostly blank, so I learnt how to use Lucene query syntax to filter out the ones without ticker names:
```"query": "formType:\"10-K\" AND ticker:[A TO Z]"```

After obtaining the text, I saved them in seperate .txt files in the data folder with the ticker name as the file name (ie. 'AADI.txt')

### Data Selection (_remove_icky_bits(), processor.py_)
After looking through some of the files generated from the data scraping process, I found that there are parts in Item 1 that do not contribute towards describing the company, and decided to remove them over three levels:
1. **paragraph / section-level** (seperated by \n)
   - table data and section headings (ie. _"(a) Our Business"_) usually show up as single lines with very few words, so I removed all lines with fewer than a threshold level of words
   - paragraphs with regulatory information / additional information (ie. _"Our website can be found here with our form 10-K and form 10-Q"_) are found and deleted with keyword searching:
       ```any(keyword in line for keyword in ['Schedule', 'Rule', 'Form', 'Section', 'www.'])```
   - the COVID-19 section (discusses the effects Covid has had on their business) is also filtered out with a string search:
      ```any(keyword in line.lower() for keyword in ['covid', 'covid-19', 'coronavirus'])```
2. **sentence-level** (seperated by full stops)
   - discussions about stock dividends / revenues exist as sentences (sometimes in paragraphs / sections that are important, so I only remove sentences with them instead of the entire pararaph / section, again with a keyword search:
   ```if any(keyword in line for keyword in ['shareholder', 'shareholders', 'stockholder', 'stockholders', 'stock']):```
3. **word-level** (seperated by space character)
   - there are symbols that are badly encoded (ie. \&\#x201d; \&\#x2019;): locate and remove them with a regex search ```re.sub('&#x?[0-9]*[a-z]?;', '', word)```
   - all acryonyms (```word.isupper()```)are removed as well so that they do not represent words that they do not mean when converted to lowercase later
   

### Data Cleaning _(clean_data(), processor.py))_
After removing the undesired parts of the text, it is time to clean the text:
1. use spacy's NER to identify names of organizations / countries / people (labels 'ORG', 'GPE', 'PERSON'), then make them one single word (the United States -> theUnitedStates), like this:
   > 'Further, due to DHE’s low oral bioavailability, there are no approved oral  products in theUnitedStates'
2. use spacy again to lemmaize words (taking into consideration parts of speech)
   > ['far', ',', 'due', 'to', 'DHE', '’s', 'low', 'oral', 'bioavailability', ',', 'there', 'be', 'no', 'approved', 'oral', ' ', 'product', 'in', 'theUnitedStates']
3. filter out numbers and punctuation from the tagged parts-of-speech from the previous section
4. remove stopwords
   > ['far', ',', 'DHE', 'low', 'oral', 'bioavailability', ',', 'approved', 'oral', ' ', 'product', 'theUnitedStates']
5. convert everything to lowercase!

<details>
  <summary>Here's an exerpt of what a document looks like past this stage!</summary>
  
  ##### GNOG.txt
  expressly state context require term company   refer goldennuggetonlinegame online gaming igaming digital sport entertainment company focus provide customer enjoyable realistic exciting online gaming experience market. currently operate newjersey michigan westvirginia offer patron ability play favorite casino game bet live action sport event virginia currently offer online sport bet. desire innovate improve offer realistic online gaming platform drive employee define business pursue vision lead destination online gaming player modern mindset. online gaming operator enter newjersey market michigan market january recently enter westvirginia market september virginia market offer online sport bet september. affiliate thegoldennugget / landry family company refer   aspire live reputation goldennugget brand storied brand gaming industry provide customer online gaming experience consistent land base casino goldennugget   nevada limited liability company indirect wholly subsidiary   define goldennugget. technology design create superior online bet experience avid casino sport bettor. goal shape player mind today anticipate gaming industry evolve. 
</details>

### Building the Model (_textpredictor.py_)
#### Vectorization
To convert text into numbers (that ML models can understand), I need to do vectorisation. I have chosen to use the TF-IDF vectoriser, as I believe its method of determining important words are useful to my case (ie. by putting less importance on words that are simply part of the standard Form 10-K format / discussions). It uses two concepts: term frequency and (inverse) document frequency, which favors terms with high frequency in a document, and are specific to that document (ie. aren't very frequent in other documents too).

#### Dimensionality Reduction
After checking the resultant array (calling ```.shape``` on the results from ```tf_transformer.transform()```), I noticed it has 26k+ features, which is _a lot_, and is bad for clustering models like Kmeans. I want to find a way to reduce the number of features. I used sklearn's ```VarianceThreshold()```, which will remove all features with variance below a threshold (which I set to be 0.0002.) After performing dimensionality reduction, 670 features are left.

#### K-means clustering
Next, comes the most important step: fitting the data into a K-means clustering model, where cluster centers are identified! I chose to keep the default number of clusters (8) because it seemed like a reasonable amount of clusters given the use case (ie. company industries).

### Results
The results sort of make sense--especially for group 1, which mainly consists of biopharmaceutical companies, and group 3, which mainly consists of blank check companies. But there are groups, like 4 and 6, that don't really seem to be grouped together well.

Here is what I think each category represents:
```
0: Tech analytics (STER, CWAN, PFMT)
1: Biomedical (ie. ATNX)
2. Finance / Investments (ie. LFT, PFIS)
3. Blank Check (ie. FCAX, GTAC)
4. Marketing (ATER, GIC) & Electricity / Lighting (EFOI, WTT) (a bit of a mess too)
5. Cannabis Distribution (ie. INSD, ITHUF) (????) 
6. Miscalleneous (a flaming hot mess)
7. Finance / Banking (ie. EMCF, RVRF)
```
<details>
<summary>See the full results here</summary>

```Group 0: STER, CWAN, PFMT, ATCX, TISI
Group 1: SYBX, SQZ, BXRX, LIFE, IDYA, PRLD, ACXP, BDTX, EFTR, IKNA, RLYB, CPRX, DBTX, AVRO, CGEM, RVPH, AADI, ACET, SYN, VICP, SANA, ETON, TCON, ATHX, MRKR, APRE, AVDL, SMMT, STSA, VIRX, CUE, CABA, IPSC, FUSN, VIRI, ACOR, ALDX, ATNX, CYT, INZY
Group 2: LFT, PFIS, CFBK, FOA, HMPT, FSBW, FGBI, CMTG
Group 3: HPLT, CBRG, CNDB, FCAX, FTAA, WNNR, OTEC, SCLE, ACII, SAMA, NVAC, BNIX, APGB, NAAC, FTPA, GTAC, CONX, HZON, GLBL, MOTV
Group 4: BIRD, EFOI, PKOH, JAKK, ATER, BIOL, WTT, GIC, TTCF, SOVO, AEIS, RTSL, BKTI, THRN, SMTC, TGLS, PWFL, IRIX, SUMR
Group 5: INSD, ITHUF, TRSSF, MILC, MRMD, NLCP
Group 6: DH, NLAB, SGC, GNOG, HDSN, CMCT, HMTV, SGA, DFH, MDRR, GJCU, ALDA, UONE, HWIN, GNE, BLNK, LEGH, PRCH, FF, CLPR, QUBT, HQI, MIMO, CRWD, RMBL, CTOS, SDSYA, CMAX, HALL, DM, SRG, INTZ, SFT, HIL, VERX, EVC, KBLB, PRPL, AP, ASZ, BMBL, BJ, LGTO, GPP, REI, BBQ, DXPE, PEI, ARSN, PLBY, KRBF, XBIT, SVNA, WHLM, NBEV, NWPX, CHMI, ARGO, RDI, LMB, TSQ, RYAN, MNTK, FLL, AC, PESI, EDR, ULH, ALTD, LSEA, HTIA, VHC, ML, AKU, PPC, GOCO, CTKB, TLIS, LOTZ, XELA, SOYB, DMS, TIG, BBXIA, BTBD, ALAC, BOX, WLMS, CPSS, INRE, BRCC, BURL, LOV, PUBC, INPX, HLMN, SPPI, ATLC, AXTI, MYSZ, MCG, FNHC, ELA, HYRE, SRGA, COUP, JOAN, BRT, STKS, RIOT, AUS, KODK, ATNI, HFFG, FNRN, NVDA, DCAC, ICD, GBLI, SIG, BRDS, EWCZ, CCOB, SPIN, GLG, DS, PFSW, CELH, NUVR, PANL, STGW, INUV, GRIL, WRBY
Group 7: BMRC, OVBC, CSBB, RVRF, EMCF, TCBX, CBAN, JUVF, COFS, MPB, UBCP, BFC, FCCO, INBK, FMCB, MRBK
```
</details>

There also seems to be an imbalance across the categories: group 6 is highly populated, followed by group 1, while the other groups have very few members. One possible cause is that there are more types of companies than the 8 clusters the model is automatically looking for, resulting in the model being forced to group many different companies into one big cluster. On the other hand, groups 2 and 7 both seem to be financial companies (and could have been merged into one group), which would imply the opposite: that there are actually less clusters than what the model is told to look for, so that it over-splits groups. 

Additionally, the groups seem to be weird, in that some of them are very specific -- there's a category specifically for cannabis distribution(!!), but there's also a rather broad categories like finance (groups 2/7). This could potentially be due to there being more companies of certain specific types in my dataset due to a selection bias when scraping my data (see last point in the Evalation section).

### Conclusion / Evaluation
Despite my model giving very weird results, given the time constraint, I'm pretty happy with my work (and learnt a lot in the process too!)
#### Thoughts for further improvement:
- I could use a vectorizer that takes into account synonyms / word meaning, or maybe find a way to merge similar words before putting it through the tf-idf vectoriser, which will serve to further reduce the dimensionality and also preserve the meaning of words. 
- I could further filter out the text used in the data selection step to better filter out noise / not useful parts of the text (maybe through selecting only sections with specific titles (ie. "Our Business"))
- I could try to see if reducing the input text to only nouns and adjectives (using scipy's part of speech tagging) could give way to better results.
  - one concern with this normally (ie. if used for sentiment analysis) is that the context of such words will be lost (ie. losing words such as "not" would reverse the meaning), but in clustering, capturing the topics discussed (through nouns & adjectives) should be sufficient so this isn't a problem!
- I could do more testing on different parameters (ie. variance threshold for dimensionality reduction, number of clusters for Kmeans model etc.)
- I could try scraping for documents over multiple months, as there could be a selection bias in my data since my dataset consists of the most recent (as of Mar 18) Form 10-K filings. If certain types of companies are more likely to release their filings on a particular date, my data will not be representative of the actual market distribution of companies.
