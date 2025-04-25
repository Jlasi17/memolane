// src/components/PatientRegistrationForm/index.jsx
import './styles.css';
import { useState } from 'react';

const PatientRegistrationForm = ({ onSubmit }) => {
  const [formData, setFormData] = useState({
    name: '',
    age: '',
    gender: '',
    medical_history: '',
    contact_info: {
      email: '',
      phone: ''
    }
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (name.includes('contact_info.')) {
      const field = name.split('.')[1];
      setFormData(prev => ({
        ...prev,
        contact_info: {
          ...prev.contact_info,
          [field]: value
        }
      }));
    } else {
      setFormData(prev => ({
        ...prev,
        [name]: value
      }));
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form className="patient-registration-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label>Full Name</label>
        <input
          type="text"
          name="name"
          value={formData.name}
          onChange={handleChange}
          required
        />
      </div>
      
      <div className="form-row">
        <div className="form-group">
          <label>Age</label>
          <input
            type="number"
            name="age"
            value={formData.age}
            onChange={handleChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label>Gender</label>
          <select
            name="gender"
            value={formData.gender}
            onChange={handleChange}
            required
          >
            <option value="">Select</option>
            <option value="male">Male</option>
            <option value="female">Female</option>
            <option value="other">Other</option>
          </select>
        </div>
      </div>
      
      <div className="form-group">
        <label>Medical History (optional)</label>
        <textarea
          name="medical_history"
          value={formData.medical_history}
          onChange={handleChange}
          rows="3"
        />
      </div>
      
      <div className="form-group">
        <label>Email</label>
        <input
          type="email"
          name="contact_info.email"
          value={formData.contact_info.email}
          onChange={handleChange}
          required
        />
      </div>
      
      <div className="form-group">
        <label>Phone Number</label>
        <input
          type="tel"
          name="contact_info.phone"
          value={formData.contact_info.phone}
          onChange={handleChange}
          required
        />
      </div>
      
      <button type="submit" className="submit-btn">
        Register Patient
      </button>
    </form>
  );
};

export default PatientRegistrationForm;