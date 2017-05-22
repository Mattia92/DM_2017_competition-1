import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class SequentialCoveringClassifier(BaseEstimator, ClassifierMixin):
	def __init__(self, demo_param='demo'):
		self.demo_param = demo_param

	def print_clause(self, clause):
		print('Rel. Accuracy: {:.2f}, TP: {}, FP: {}, Col: {} -> ({:.2f}, {:.2f}]'.format(clause[0], clause[1], clause[2], clause[3], clause[4], clause[5]))

	def find_best_rule(self, X1, y1, k=4, min_coverage=100, acc_imp_tresh=0.01, max_clauses=10, min_accuracy=0.5):
		y1 = y1.copy()
		X1 = X1.copy()
		rule = []
		current_clause = 0
		print('Starting find_best_rule on {} samples, {} positives'.format(y1.count(), y1.sum()))
		while (True):
			y1_count = y1.count()
			best_clause = (0, 0, 0, 0, 0)
			rule_found = False
			for col in X1.columns:
				X1_col = X1[col]
				lb = X1_col.min()
				ub = X1_col.max()
				print('col = {}, k = {}, lb = {:.2f}, ub = {:.2f}'.format(col, k, lb, ub))
				for i in np.arange(2, k+1):
					h = (ub - lb) / i
					# print('h = {:.2f}'.format(h))
					for j in np.arange(0, i):
						base = lb + j * h
						top = base + h
						# print('base = {:.2f}, top = {:.2f}'.format(base, top))
						mask = (X1_col > base) & (X1_col <= top)
						# X2 = X1_col[mask]
						y2 = y1[mask]
						y2_count = y2.count()
						if (y2_count >= min_coverage):
							y2_sum = y2.sum()
							acc = y2_sum / y2_count
							if ((current_clause == 0 or (current_clause >= 1 and acc > rule[current_clause - 1][0] + acc_imp_tresh)) and
									acc > min_accuracy and
									acc > best_clause[0]):
								rule_found = True
								best_clause = (acc, y2_sum, y2_count - y2_sum, col, base, top)
								print('New best clause found!')
								self.print_clause(best_clause)
			if (not rule_found):
				break
			# Add clause to rule
			rule.append(best_clause)
			# Compute mask to remove positive samples covered by the added clause
			mask = self.mask(best_clause, X1)
			X1 = X1[mask]
			y1 = y1[mask]
			# Add 1 to the clause counter
			current_clause += 1
			# Exit if clause counter too high
			if (current_clause >= max_clauses):
				break
		print('Best clause found so far:')
		for clause in rule:
			self.print_clause(clause)
		return rule

	def mask(self, clause, X1):
		return (X1[clause[3]] > clause[4]) & (X1[clause[3]] <= clause[5])

	def cover(self, rule, X1):
		covered = pd.DataFrame(index=X1.index)
		covered['coverage'] = True
		for clause in rule:
			covered['coverage'] &= self.mask(clause, X1)
		# print(covered['coverage'])
		return covered['coverage']	

	def post_process(self, rule_list):
		# TODO
		return rule_list

	def fit(self, X, y):
		# Check that X and y have correct shape
		# X, y = check_X_y(X, y)
		
		# Store the classes seen during fit
		self.classes_ = unique_labels(y)

		self.X_ = X
		self.y_ = y

		# Rule list is initially empty
		rule_list = []

		X1 = pd.DataFrame(X)
		y1 = pd.DataFrame(y)['DEFAULT PAYMENT JAN'].apply(lambda x: True if x==1 else False)
		
		while (not y1.empty):
			# Find the best rule possible
			rule = self.find_best_rule(X1, y1)
			# Check if we need more rules
			if len(rule) == 0:
				print('Stopping, there are {} examples that are going to be predicted using majority class'.format(y1.count()))
				print('Rules mined: {}'.format(len(rule_list)))
				for i in np.arange(0, len(rule_list)):
					print('Rule {}/{}'.format(i+1,len(rule_list)))
					for clause in rule_list[i]:
						self.print_clause(clause)
				break
			# Remove covered examples and update the moodel
			mask = ~self.cover(rule, X1)
			X1 = X1[mask]
			y1 = y1[mask]

			rule_list.append(rule)

		# Post-process the rules (sort-them, simplify them, etc.)
		rule_list = self.post_process(rule_list)

		self.rule_list = rule_list

		# Return the classifier
		return self

	def predict(self, X):
		# Check is fit had been called
		check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		X = check_array(X)

		def check_rules(x):
			for rule in rule_list:
				if (cover(rule, x)):
					return True
			return False

		y = X.apply(lambda x: check_rules(x))

		# y_pred = 
#
		return y

		# closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
		# return self.y_[closest]
