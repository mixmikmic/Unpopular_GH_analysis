import pandas as pd
import numpy as np

thumb_imgs_long = pd.read_csv("../assets/02_thumb_text_data.csv")
media_imgs_long = pd.read_csv("../assets/02_media_text_data.csv")

thumb_map = pd.read_csv("../assets/mapping_dict_thumbnail.csv")
media_map = pd.read_csv("../assets/mapping_dict_mediaurl.csv")

thumb_map.columns = ["img", "url"]
media_map.columns = ["img", "url"]

thumb = thumb_imgs_long.drop("Unnamed: 0", axis =1)
thumb = thumb.merge(thumb_map, on = "img")

media = media_imgs_long.drop("Unnamed: 0", axis = 1)
media = media.merge(media_map, on="img")

thumb[thumb.isnull().any(axis=1)]

media.head()

thumb.columns = ["thumb-celeb","thumb-img","thumb-text", "0", "1", "2", "3", "4", "5", "6", "7", "8", 
                 "9", "10", "11", "12", "13", "link_thumbnail"]

media.columns = ["media-celeb","media-img","media-text", "0", "1", "2", "3", "4", "5", "6", "7", "8", 
                 "9", "10", "11", "12", "13", "media_url"]

all_data = pd.read_csv("../gitignore/newtweets_10percent.csv")

all_data.columns

all_data = all_data[["id", "brand", "link_thumbnail", "media_url", "engagement", "impact", 
                    "timestamp", "hashtags", "favorite_count", "retweet_count", "text", "tweet_url"]]

all_data = all_data.merge(thumb, on = "link_thumbnail", how = "outer")

all_data = all_data.merge(media, on = "media_url", how = "outer")

# x is thumbnail, y is media. will be working to condense this
all_data

all_data.columns

#null thumbnails
len(all_data[all_data["link_thumbnail"].isnull()])
thumbnail_i = 10000-len(all_data[all_data["link_thumbnail"].isnull()])

thumbnail_i

#null media
len(all_data[all_data["media_url"].isnull()])
media_i = 10000-len(all_data[all_data["media_url"].isnull()])

media_i

len(all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].isnull())])

#800 records have either thumbnail or media link.

len(all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())])

#^^ records have both. I inspected these and discovered they are the same image. When merging, will favor thumbnail.

#confirming that if an image has both thumbnail and media link, they are the same image seen in tweet.
list(all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())]["link_thumbnail"])

new_df = all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].isnull())]

new_df.columns

new_df = new_df[["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]

new_df.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

len(new_df)

new_df.info()

new_df.head()

thumb_only = all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].isnull())]

thumb_only = thumb_only[["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]

thumb_only.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

new_df = new_df.append(thumb_only)

len(new_df)

sum(new_df.duplicated())

#total number of tweets with no image shown despite a thumbnail link
5597-5034-200

cleanup_thumb = new_df[new_df["image_url"].notnull() & new_df["thumb_img"].isnull()][["id", "image_url"]]

cleanup_thumb.to_csv("cleanup_thumb.csv")

new_df.info()

media_only = all_data[(all_data["link_thumbnail"].isnull())&(all_data["media_url"].notnull())]

media_only = media_only[["id", "brand", "timestamp", "media_url", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "media-img", "media-celeb", 
                 "media-text", "0_y", "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", 
                 "10_y", "11_y", "12_y", "13_y"]]

media_only.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

media_cleanup = media_only[media_only["media_img"].isnull()][["id", "image_url"]]

media_cleanup.to_csv("cleanup_media.csv")

new_df = new_df.append(media_only)

len(new_df)

sum(new_df.duplicated())

new_df = new_df.drop_duplicates()

len(new_df)

5910 - 5333

new_df.info()

both = all_data[(all_data["link_thumbnail"].notnull())&(all_data["media_url"].notnull())]

len(both)

list(both["link_thumbnail"])[9]

list(both["tweet_url"])[9]

len(both)

both[["tweet_url", "link_thumbnail", "media_url","media-celeb", "thumb-img", "media-img", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", "10_x", "11_x", "12_x", "13_x", "0_y", 
      "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", "10_y", "11_y", "12_y", "13_y"]]

both_thumb = both[(both["thumb-img"].notnull())&(both["media-img"].isnull())][["id", "brand", "timestamp", "link_thumbnail", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "thumb-img", "thumb-img", "thumb-celeb", 
                 "thumb-text", "0_x", "1_x", "2_x", "3_x", "4_x", "5_x", "6_x", "7_x", "8_x", "9_x", 
                 "10_x", "11_x", "12_x", "13_x"]]

both_thumb.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

both_media = both[(both["media-img"].notnull())&(both["thumb-img"].isnull())][["id", "brand", "timestamp", "media_url", "engagement", "impact", "favorite_count", 
                 "retweet_count", "text", "hashtags", "tweet_url", "media-img", "media-img", "media-celeb", 
                 "media-text", "0_y", "1_y", "2_y", "3_y", "4_y", "5_y", "6_y", "7_y", "8_y", "9_y", 
                 "10_y", "11_y", "12_y", "13_y"]]

both_media.columns = ["id", "brand", "timestamp", "image_url", "engagement", "impact", "favorite_count", 
                  "retweet_count", "text", "hashtags", "tweet_url", "thumb_img", "media_img", "img_celeb", "img_text",
                  "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]

reduced_both = both_thumb.append(both_media)

len(reduced_both)

reduced_both.info()

new_df = new_df.append(reduced_both)

len(new_df)



