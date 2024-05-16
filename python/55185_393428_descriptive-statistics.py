import pandas as pd
import numpy as np
from scipy import stats

#from qtextasdata import QTextAsData,QInputParams
# def query_database_harelba_q():
#     # Create an instance of q. Default input parameters can be provided here if needed
#     q = QTextAsData()

#     # execute a query, using specific input parameters
#     r = q.execute('select * from /etc/passwd',QInputParams(delimiter=':'))

#     # Get the result status (ok/error). In case of error, r.error will contain a QError instance with the error information
#     print r.status

sessions = pd.read_csv('./data/sessions-hypercube.csv')
sessions_orig = pd.read_csv('./data/sessions-with-features.csv')
orig_sessions_female_control = sessions_orig.loc[sessions_orig['gender'] == 'female'].loc[sessions_orig['variant']== 'control']
print sessions_orig.head(), len(sessions_orig)
print orig_sessions_female_control.head(), len(orig_sessions_female_control)

#female_sessions = sessions.loc[sessions['gender'] == 'female']
female_sessions_control = sessions.loc[sessions['gender'] == 'female'].loc[sessions['variant']== 'control']
female_sessions_test = sessions.loc[sessions['gender'] == 'female'].loc[sessions['variant']== 'test']
#print female_sessions_control.head(), len(female_sessions_control)

rps_female_ctrl = np.divide(female_sessions_control.rps_sum, female_sessions_control.n)
type(pd.DataFrame(rps_female_ctrl))
rps_female_ctrl = np.divide(female_sessions_control.rps_sum, female_sessions_control.n)
type(pd.DataFrame(rps_female_ctrl))

female_sessions_control['mean_rps'] = female_sessions_control.rps_sum/female_sessions_control.n 
female_sessions_test['mean_rps'] = female_sessions_test.rps_sum/female_sessions_test.n 

print female_sessions_control.head(), type(female_sessions_control)

print "\nrps_ctrl: ", female_sessions_control['mean_rps'].mean()
print "rps_test: ",female_sessions_test['mean_rps'].mean()

print "rps_diff: ", np.abs(female_sessions_control['mean_rps'].mean() 
                           - female_sessions_test['mean_rps'].mean())

print "rps_diff_frac: ", np.divide(female_sessions_test['mean_rps'].mean(),
                        female_sessions_control['mean_rps'].mean())

print "rps_diff_err: ", stats.sem(orig_sessions_female_control['rps'])




