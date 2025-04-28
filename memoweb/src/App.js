// src/App.js
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import AuthPage from './components/Auth/AuthPage';
import FamilyHome from './components/pages/FamilyHome';
import LoginForm from './components/Auth/LoginForm';
import DoctorHome from './components/pages/DoctorHome';
import MemoTap from './components/games/memotap';
import PatientHome from './components/pages/PatientHome';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<AuthPage />} />
        <Route path="/family" element={<FamilyHome />} />
        <Route path="/doctor" element={<DoctorHome />} />
        <Route path="/login" element={<LoginForm />} />
         <Route path="/memotap" element={<MemoTap />} /> 
         <Route path="/patient" element={<PatientHome />} />
      </Routes>
    </Router>
  );
}

export default App;