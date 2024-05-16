import praw

# I saved my user agent string in a local file
# since one should use their own
with open('./.user_agent') as f:
    user_agent = f.read()

# instantiate Reddit connection class
r = praw.Reddit(user_agent=user_agent)

# let's get the current top 10 submissions
# since praw interacts with reddit lazily, we wrap the method
# call in a list
submissions = list(r.get_subreddit('apple').get_hot(limit=10))
[str(s) for s in submissions]

submissions = submissions[2:]
[str(s) for s in submissions]

# grab the first submission
submission = submissions[0]

# the actual text is in the body
# attribute of a Comment
def get_comments(submission, n=20):
    """
    Return a list of comments from a submission.
    
    We can't just use submission.comments, because some
    of those comments may be MoreComments classes and those don't
    have bodies.
    
    n is the number of comments we want to limit ourselves to
    from the submission.
    """
    count = 0
    def barf_comments(iterable=submission.comments):
        """
        This generator barfs out comments.
        
        Some comments seem not to have bodies, so we skip those.
        """
        nonlocal count
        for c in iterable:
            if hasattr(c, 'body') and count < n:
                count += 1
                yield c.body
            elif hasattr(c, '__iter__'):
                # handle MoreComments classes
                yield from barf_comments(c)
            else:
                # c was a Comment and did not have a body
                continue
    return barf_comments()
                
comments = list(get_comments(submission))
list(comments)

from textblob import TextBlob

comment_blob = TextBlob(''.join(comments))

more_comments = []

for submission in submissions:
    more_comments.extend(get_comments(submission, 200))

len(more_comments)

bigger_blob = TextBlob(''.join(more_comments))

# The first time I ran this method, it failed
# because I hadn't read TextBlob's docs closely
# and downloaded the corpus of text in needed.
# python -m textblob.download_corpora

print(len(bigger_blob.words))

from collections import Counter

counter = Counter(bigger_blob.words)

# the most common words are pretty mundane common parts of speech, so we'll skip the first few
counter.most_common()[60:100]

bigger_blob.sentiment



