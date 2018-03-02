import re
import urllib.request
import urllib.parse
import urllib.error

pages = ['beethoven.htm', 'bach.htm', ]
base = 'http://classicalmidi.co.uk/'

# https://stackoverflow.com/questions/4389572/how-to-fetch-a-non-ascii-url-with-python-urlopen
def urlEncodeNonAscii(b):
    return re.sub('[\x80-\xFF]', lambda c: '%%%02x' % ord(c.group(0)), b)
def iriToUri(iri):
    parts= urllib.parse.urlparse(iri)
    return urllib.parse.urlunparse(
        part.encode('idna') if parti==1 else urlEncodeNonAscii(part.encode('utf-16'))
        for parti, part in enumerate(parts)
    )

for page in pages:
    with urllib.request.urlopen(base + page) as response:
       html = response.read().decode('windows-1252')

       matches = re.findall(r'<a href="(.+\.mid)">', html, re.IGNORECASE)
       for match in matches:
            if match[0] == '/':
                match = match[1:]

            url = base + match
            url = urllib.parse.quote(url.encode('utf8'), ':/')
            try:
                urllib.request.urlretrieve(url, './input/' + match.split('/')[-1])
                print('succeeded on {}'.format(url))
            except urllib.error.HTTPError:
                print('failed on {}'.format(url))



