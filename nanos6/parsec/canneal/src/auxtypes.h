#ifndef _AUXTYPES_H
#define _AUXTYPES_H

#include <cstddef>
#include <utility>
#include <string>
#include <iostream>

template <class T>
class AuxVector {
	protected:
		T *_data;
		int _size;
		int _dim;

	public:
		AuxVector(): _data(NULL), _size(0), _dim(0) {}
		AuxVector(int s): _size(0), _dim(s) {_data = new T[s];}
		~AuxVector() {if(_data) delete[] _data;}
		int size() {return _size;}
		T &operator[](int i) {return _data[i];}
		const T &operator[](int i) const {return _data[i];}
		T &at(int i) {return _data[i];}
		const T &at(int i) const {return _data[i];}
		void push_back(const T &val) {_data[_size++] = val;}
		void resize(int s, T val) {
			if(_dim < s) redim(s);
			for (int i = _size; i < s; ++i)
				_data[i] = val;
			_size = s;
		}
		void resize(int s) {
			if(_dim < s) redim(s);
			_size = s;
		}
		void redim(int s) {
			T *auxp = new T[s];
			if (_data) {
				for (int i = 0; i < _size; ++i)
					auxp[i] = _data[i];
				delete[] _data;
			}
			_data = auxp;
		}
		AuxVector<T> &operator= (const AuxVector<T> &x) {
			_size = x._size;
			_dim = x._dim;
			if (_data) delete[] _data;
			_data = new T[_dim];
			for (int i = 0; i < _size; ++i)
				_data[i] = x._data[i];
			return *this;
		}
};

template <class K, class T>
class AuxMap: public AuxVector< std::pair<K, T> > {
	public:
		int find(const K &k) {
			for (int i = 0; i < this->_size; ++i) {
				if (this->_data[i].first == k) return i;
			}
			return -1;
		}
		T &operator[](const K &k) {
			int i = find(k);
			if (i == -1) {
				push_back(std::pair<K, T>());
				this->_data[this->_size-1].first = k;
				return this->_data[this->_size-1].second;
			}
			return this->_data[i].second;
		}
		const T &operator[](const K &k) const {
			int i = find(k);
			if (i == -1) {
				push_back(std::pair<K, T>());
				this->_data[this->_size-1].first = k;
				return this->_data[this->_size-1].second;
			}
			return this->_data[i].second;
		}
		
};

template <class T>
class AuxHash: public AuxMap<std::string, T> {
	private:
		int hash_func(const std::string &k) {
			int ind = 0;
			for (int i = 0; i < k.size(); ++i) {
				ind *= 27;
				ind += k[i] - 'a' + 1;
			}
			return ind;
		}
	public:
		int find(const std::string &k) {
			int i;
			for (i = hash_func(k); this->_data[i].first != ""; i = (i+1)%(this->_dim)) {
				if (this->_data[i].first == k) return i;
			}
			return -i;
		}
		T &operator[](const std::string &k) {
			int i = find(k);
			if (i < 0) {
				i = -i;
				this->_data[i].first = k;
			}
			//else std::cerr << "Existent" << std::endl;
			return this->_data[i].second;
		}
		const T &operator[](const std::string &k) const {
			int i = find(k);
			if (i < 0) {
				i = -i;
				this->_data[this->_size-1].first = k;
			}
			return this->_data[i].second;
		}
		
};

#endif
