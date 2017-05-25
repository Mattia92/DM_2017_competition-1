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
		print('Rel. Accuracy: {:.2f}, TP: {}, FP: {}, Col: {} -> ({:.2f}, {:.2f}]'.format(
			clause[0], clause[1], clause[2], clause[3], clause[4], clause[5]))

	def find_best_rule(self, X, y, k=8, min_coverage=100,
		acc_imp_tresh=0.01, max_clauses=8, min_accuracy=0.23):
		y = pd.Series(y)
		X = pd.DataFrame(X)
		rule = []
		current_clause = 0
		best_clause = (0, 0, 0, 0, 0)
		print('Starting find_best_rule on {} samples, {} positives'.format(
			y.count(), y.sum()))
		while (True):
			y_count = y.count()
			rule_found = False
			for col in X.columns:
				X_col = X[col]
				lb = X_col.min()
				ub = X_col.max()
				# print('col = {}, k = {}, lb = {:.2f}, ub = {:.2f}'.format(col, k, lb, ub))
				for i in np.arange(2, k+1):
					h = (ub - lb) / i
					# print('h = {:.2f}'.format(h))
					for j in np.arange(0, i):
						base = lb + j * h
						top = base + h
						# print('base = {:.2f}, top = {:.2f}'.format(base, top))
						mask = (X_col > base) & (X_col <= top)
						# X2 = X1_col[mask]
						y_masked = y[mask]
						y_masked_count = y_masked.count()
						if (y_masked_count >= min_coverage):
							y_masked_sum = y_masked.sum()
							acc = y_masked_sum / y_masked_count
							if ((acc > best_clause[0] + acc_imp_tresh) or (acc > best_clause[0] - acc_imp_tresh and y_masked_sum > best_clause[1])):
								rule_found = True
								best_clause = (acc, y_masked_sum, y_masked_count - y_masked_sum, col, base, top)
								# print('New best clause found!')
								# self.print_clause(best_clause)
			if (not rule_found):
				break
			if (best_clause[0] < min_accuracy):
				break
			# Add clause to rule
			rule.append(best_clause)
			# Compute mask to remove positive samples covered by the added clause
			mask = self.mask(best_clause, X)
			X = X[mask]
			y = y[mask]
			# Add 1 to the clause counter
			current_clause += 1
			# Exit if clause counter too high
			if (current_clause >= max_clauses):
				break
		print('Best clause found so far:')
		for clause in rule:
			self.print_clause(clause)
		return rule

	def mask(self, clause, X):
		return (X[clause[3]] > clause[4]) & (X[clause[3]] <= clause[5])

	def cover(self, rule, X):
		covered = pd.Series(True, index=X.index)
		for clause in rule:
			covered &= self.mask(clause, X)
		#print(covered)
		return covered

	def post_process(self, rule_list):
		for rule in rule_list:
			len_rule = len(rule)
			i = 0
			while (i < len_rule - 1):
				j = i + 1
				while (j < len_rule):
					# Being futher is more precise, thus remove the previous
					if (rule[i][3] == rule[j][3]):
						rule.remove(rule[i])
						len_rule -= 1
					j += 1
				i += 1
		return rule_list

	def fit(self, X, y):
		# Check that X and y have correct shape
		# X, y = check_X_y(X, y)
		
		# Store the classes seen during fit
		# self.classes_ = unique_labels(y)

		# self.X_ = X
		# self.y_ = y

		# Rule list is initially empty
		rule_list = []

		X = pd.DataFrame(X)
		y = pd.Series(y).apply(lambda x: x == 1)
		
		while (not y.empty):
			# Find the best rule possible
			rule = self.find_best_rule(X, y)
			# Check if we need more rules
			if len(rule) == 0:
				print('Stopping, there are {} examples that are going to be predicted using majority class'.format(y.count()))
				break
			# Remove covered examples and update the moodel
			mask = ~self.cover(rule, X)
			X = X[mask]
			y = y[mask]

			rule_list.append(rule)
			# input()

		# Post-process the rules (sort-them, simplify them, etc.)
		self.rule_list = self.post_process(rule_list)

		print('Rules mined: {}'.format(len(rule_list)))
		for i in np.arange(0, len(rule_list)):
			print('Rule {}/{}'.format(i+1,len(rule_list)))
			for clause in rule_list[i]:
				self.print_clause(clause)

		# Return the classifier
		return self

	def predict(self, X):
		# Check is fit had been called
		# check_is_fitted(self, ['X_', 'y_'])

		# Input validation
		# X = check_array(X)

		y_pred = pd.Series(False, index=X.index)
		for rule in self.rule_list:
			#print(self.cover(rule, X))
			y_pred |= self.cover(rule, X)

		#print(y_pred.sum())

		y_pred.apply(lambda x: 1 if x else 0)

		return y_pred

		# closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
		# return self.y_[closest]
