import axios from 'axios';

const API = axios.create({
  baseURL: 'http://localhost:8000'
});

export const login = async (credentials) => {
  const response = await API.post('/login', credentials);
  return response.data;
};

export const register = async (userData) => {
  const response = await API.post('/register', userData);
  return response.data;
};