import os
import re

in_directory = './jobs_text_complete/affirm/'
out_directory = './jobs_text_meat_only/affirm/'

doc_num = 0

def extract_meat(in_directory, filename):
    meat = []
    bad_lines = ['what you\'ll do', 'what we look for', 'who we look for']
    abandon = ['about affirm', 'apply for this job', 'at affirm we are using technology to re-imagine and re-build core parts of financial infrastructure to enable cheaper, friendlier, and more transparent financial products and services that improve lives.']

    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'")
            counter += 1
            if counter in [1,2,3,4]:
                continue
            if counter == 0:
                title = line
            elif line in bad_lines:
                continue
            elif line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Affirm ' + title.title()
        header = 'Affirm ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ') + '.txt' 
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(directory):
    extract_meat(directory, filename)
    doc_num += 1
print("Processed", doc_num, "documents.")

about_uber = ['We’re changing the way people think about transportation. Not that long ago we were just an app to request premium black cars in a few metropolitan areas. Now we’re a part of the logistical fabric of more than 600 cities around the world. Whether it’s a ride, a sandwich, or a package, we use technology to give people what they want, when they want it.',
              ' For the people who drive with Uber, our app represents a flexible new way to earn money. For cities, we help strengthen local economies, improve access to transportation, and make streets safer.',
              ' And that’s just what we’re doing today. We’re thinking about the future, too. With teams working on autonomous trucking and self-driving cars, we’re in for the long haul. We’re reimagining how people and things move from one place to the next.',
              ' Uber is a technology company that is changing the way the world thinks about transportation. We are building technology people use everyday. Whether it\'s heading home from work, getting a meal delivered from a favorite restaurant, or a way to earn extra income, Uber is becoming part of the fabric of daily life.',
              ' We\'re making cities safer, smarter, and more connected. And we\'re doing it at a global scale-energizing local economies and bringing opportunity to millions of people around the world.',
              ' Uber\'s positive impact is tangible in the communities we operate in, and that drives us to keep moving forward.',
             'at uber, we pride ourselves on the amazing team we\'ve built. the driver behind all our growth, our bold and disruptive brand, and the game-changing technology we bring to market is the people that make uber well, uber.',
             'uber is an equal opportunity employer and enthusiastically encourages people from a wide variety of backgrounds and experiences to apply. uber does not discriminate on the basis of race, color, religion, sex (including pregnancy), gender, national origin, citizenship, age, mental or physical disability, veteran status, marital status, sexual orientation or any other basis prohibited by law.',
              'we\'re changing the way people think about transportation. not that long ago we were just an app to request premium black cars in a few metropolitan areas. now we\'re a part of the logistical fabric of more than 500 cities around the world. whether it\'s a ride, a sandwich, or a package, we use technology to give people what they want, when they want it.',
              'for the people who drive with uber, our app represents a flexible new way to earn money. for cities, we help strengthen local economies, improve access to transportation, and make streets safer.',
              'uber\'s positive impact is tangible in the communities we operate in, and that drives us to keep moving forward'
             ]

about_uber_lower = [item.strip().lower().replace('’',"'") for item in about_uber]

throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'qualifications & requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise']


uber_bad_lines = sorted(list(set(about_uber_lower + throwaway + [item + ':' for item in throwaway])))

# for item in uber_bad_lines:
#     print(item)

uber_abandon = ['perks', 'apply now', 'benefits', 'benefits:', 'perks:', 'apply now:', 'benefits (u.s.)']

in_directory = './jobs_text_complete/uber/'
out_directory = './jobs_text_meat_only/uber/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'")
            counter += 1
            if counter in range(116):
                continue
            elif line in bad_lines:
                continue
            elif line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Uber ' + title.title()
        header = 'Uber ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ') + '.txt' 
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, uber_bad_lines, uber_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


# Don't appear to be bad lines... tbd
throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role'
               ]

about_sf = ['about salesforce: salesforce, the customer success platform and world\'s #1 crm, empowers companies to connect with their customers in a whole new way. the company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model. these founding principles have taken our company to great heights, including being named one of forbes\'s “world\'s most innovative company” six years in a row and one of fortune\'s “100 best companies to work for” nine years in a row. we are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at salesforce. together, with our whole ohana (hawaiian for "family") made up of our employees, customers, partners and communities, we are working to improve the state of the world.',
           'salesforce is a critical business skill that anyone should have on their resume. by 2022, salesforce and our ecosystem of customers and partners will drive the creation of 3.3 million new jobs and more than $859 billion in new business revenues worldwide according to idc. salesforce is proud to partner with deloitte and our entire community of trailblazers to build a bridge into the salesforce ecosystem. our pathfinder program provides the training and accreditation necessary to be positioned for high paying jobs as salesforce administrators and salesforce developers.',
            'salesforce, the customer success platform and world\'s #1 crm, empowers companies to connect with their customers in a whole new way. the company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model. these founding principles have taken our company to great heights, including being named forbes\' \"world\'s most innovative company\" in 2017 and one of the \"world\'s most innovative company\" the previous five years. we have also been named one of fortune\'s \"100 best companies to work for\" nine years in a row. we are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at salesforce. together, with our whole ohana (hawaiian for \"family\") made up of our employees, customers, partners and communities, we are working to improve the state of the world.'
           ]

embedded_useless_sentences_raw = ['Salesforce, the Customer Success Platform and world\'s #1 CRM, empowers companies to connect with their customers in a whole new way.', 
                              'The company was founded on three disruptive ideas: a new technology model in cloud computing, a pay-as-you-go business model, and a new integrated corporate philanthropy model.',
                              'These founding principles have taken our company to great heights, including being named one of Forbes’s “World’s Most Innovative Company” five years in a row and one of Fortune’s “100 Best Companies to Work For” eight years in a row. We are the fastest growing of the top 10 enterprise software companies, and this level of growth equals incredible opportunities to grow a career at Salesforce.',
                              'Together, with our whole Ohana (Hawaiian for \"family\") made up of our employees, customers, partners and communities, we are working to improve the state of the world.']

embedded_useless_sentences = [item.lower() for item in embedded_useless_sentences_raw]

sf_bad_lines = sorted(list(set([item.strip().lower() for item in about_sf] + 
                               throwaway + 
                               [item + ':' for item in throwaway])))

sf_abandon = ['would you like to apply for this job?', 'would you like to apply to this job?']

in_directory = './jobs_text_complete/salesforce/'
out_directory = './jobs_text_meat_only/salesforce/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if counter in range(980):
                continue
            elif line in bad_lines:
                continue
            for item in embedded_useless_sentences:
                line_fixed = line.replace(item, '')
                line = line_fixed                
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Salesforce ' + title.title()
        header = 'Salesforce ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ')
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, sf_bad_lines, sf_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")


# Don't appear to be bad lines... tbd
throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role',
             'required skills ',
             'preferred skills ',
             'position   description',
             'position description',
             'education and training',
             'minimum required knowledge, skills, and abilities',
             'nice to haves',
             'skills & qualifications',
             'preferred skills',
             'education and experience:',
             'preferred',
             'job duties and responsibilities',
             'minimum required knowledge, skills, and abilities ',
             'skills and experience',
             'what we are looking for',
             'bonus skills',
             'education and certification'
               ]


okta_bad_lines = sorted(list(set(throwaway + 
                               [item + ':' for item in throwaway])))

okta_abandon = ['okta is an equal opportunity employer', 'okta is an equal opportunity employer.',
               'okta is an equal opportunity employer', 'okta is an equal opoortunity employer']

in_directory = './jobs_text_complete/okta/'
out_directory = './jobs_text_meat_only/okta/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = -1
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            title = filename
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if counter in range(209):
                continue
            elif line in bad_lines:
                continue             
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
            if line[:10] == 'apply now':
                break
            if 'u.s. equal opportunity employment information' in line:
                break
        formatted_title = 'Okta ' + title.title()
        header = 'Okta ' + title.title().replace('(',' ').replace(')',' ').replace('/',' and ')
        
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, okta_bad_lines, okta_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")

throwaway = ['what you\'ll do',
             '---', 
             'looking for',
             'the role', 
             'about the role & the team',
             'the company:',
             'successful candidates will bring',
             'bonus points if','what you\'ll need', 
             'about the job', 
             'about the role',
             'about the team',
             'responsibilities', 
             'you will', 
             'you have',
            'what you need',
            'who you are',
            'about',
            'the role',
            'org',
            'qualifications & requirements',
            '#li-post',
            '#ai-labs-jobs',
            'about this role',
            '(',
            'bonus points',
            'what we\'re looking for',
            'what you\'ll need',
            'at a glance',
            'about you',
            'qualifications',
            'what you\'ll experience',
            'the ideal candidate',
            'you are',
            'who you\'ll have',
            'requirements',
            'about the team',
            'job description',
            'what you\'ll need',
            'san francisco, ca',
            'the candidate(s) need to have the following skills',
            'key responsibilities:',
            'about uber',
            'skills',
            'expertise',
               'required skills/ experience',
               'desired skills/experience',
             'desired skills',
             'required skills',
             'want to help salesforce in a big way?',
             '*li-y',
             'basic requirements',
             'your impact',
             'role description',
             'preferred requirements',
             'if hired, a form i-9, employment eligibility verification, must be completed at the start of employment.  *li-y',
             '*li-y **li-sj',
             'minimum qualifications',
             'about you…',
             'top 5 reasons to join the team',
             'you are/have',
             'required skills/experience',
             'although the following are not required, they are considered a significant plus for this role',
             'role summary',
             'skills desired',
             'skills required',
             'skills and experience necessary for this role',
             'required skills ',
             'preferred skills ',
             'position   description',
             'position description',
             'education and training',
             'minimum required knowledge, skills, and abilities',
             'nice to haves',
             'skills & qualifications',
             'preferred skills',
             'education and experience:',
             'preferred',
             'job duties and responsibilities',
             'minimum required knowledge, skills, and abilities ',
             'skills and experience',
             'what we are looking for',
             'bonus skills',
             'education and certification',
             'company description',
             'full-time',
             'san francisco, ca, usa',
             'bonus',
             'contract'
               ]

about_square = ['We believe everyone should be able to participate and thrive in the economy. So we’re building tools that make commerce easier and more accessible to all.  We started with a little white credit card reader but haven’t stopped there. Our new reader helps our sellers accept chip cards and NFC payments, and our Cash app lets people pay each other back instantly.   We’re empowering the independent electrician to send invoices, setting up the favorite food truck with a delivery option, helping the ice cream shop pay its employees, and giving the burgeoning coffee chain capital for a second, third, and fourth location.  Let’s shorten the distance between having an idea and making a living from it. We’re here to help sellers of all sizes start, run, and grow their business—and helping them grow their business is good business for everyone.',
                'we believe everyone should be able to participate and thrive in the economy. so we\'re building tools that make commerce easier and more accessible to all. we started with a little white credit card reader but haven\'t stopped there. our new reader helps our sellers accept chip cards and nfc payments, and our cash app lets people pay each other back instantly. we\'re empowering the independent electrician to send invoices, setting up the favorite food truck with a delivery option, helping the ice cream shop pay its employees, and giving the burgeoning coffee chain capital for a second, third, and fourth location. let\'s shorten the distance between having an idea and making a living from it. we\'re here to help sellers of all sizes start, run, and grow their business—and helping them grow their business is good business for everyone.'
               ]


square_bad_lines = sorted(list(set([item.lower() for item in about_square] +
                                   throwaway + 
                               [item + ':' for item in throwaway])))

square_abandon = ['additional information']

in_directory = './jobs_text_complete/square/'
out_directory = './jobs_text_meat_only/square/'

doc_num = 0

def extract_meat(in_directory, filename, bad_lines, abandon):
    meat = []
    counter = 0
    with open (os.path.join(in_directory, filename), 'r') as infile:
        for line in infile:
            if counter == 0:
                title = line
#             line = re.sub(r'[^\x00-\x7f]',r' ',line)   # Each char is a Unicode codepoint.
            line = line.strip().lower().replace('’',"'").replace('“','"')
            counter += 1
            if line in bad_lines:
                continue             
            if line not in abandon:
                meat.append(line)
            if line in abandon:
                break
        formatted_title = 'Square ' + title.title()
        header = 'Square ' + title.title().replace('(','-').replace(')','-').replace('/','-') + '.txt'
        # Output results to file
        with open(os.path.join(out_directory, header.replace(' ','_').replace(',','_')), 'w+') as outfile:
            outfile.write(formatted_title) # writing title as first line of each doc
            outfile.write("\n\n")
            for slab in meat:
                outfile.write(slab)
                outfile.write('\n')
                
for filename in os.listdir(in_directory):
    extract_meat(in_directory, filename, square_bad_lines, square_abandon)
    doc_num += 1
print("Processed", doc_num, "documents.")

