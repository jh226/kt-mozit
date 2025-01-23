import React from 'react';
import { Routes, Route, Navigate  } from 'react-router-dom';
import { useAuth } from '../Context/AuthContext';
import QuestionListPage from '../admin-page/QuestionListPage';
import AnswerPage from '../admin-page/AnswerPage';
import NoticeListPage from '../admin-page/NoticeListPage';
import NoticeUpdatePage from '../admin-page/NoticeUpdatePage';
import NoticeEditPage from '../admin-page/NoticeEditPage';
import NoticeCreatePage from '../admin-page/NoticeCreatePage';
import UserListPage from '../admin-page/UserListPage';
import SystemStatus from '../admin-page/SystemStatus';
import Traffic from '../admin-page/Traffic';
import KPI from '../admin-page/KPI';
import AdminEdit from '../admin-page/AdminEdit'
import DashBoard from '../dashboard/Dashboard'


const AdminRoutes = () => {
    const { userId } = useAuth();

    if (userId !== 'admin') {
        if(userId){
            alert("관리자만 접근 가능합니다.")
            return <Navigate to="/" replace />;
        }
        return <Navigate to="/sign-in" replace />;
    }

    return (
        <Routes>
            <Route path="/admin/dashboard" element={<DashBoard />} />
            <Route path="/admin/system-status" element={<SystemStatus />} />
            <Route path="/admin/traffic" element={<Traffic />} />
            <Route path="/admin/users" element={<UserListPage />} />
            <Route path="/admin/notice" element={<NoticeListPage />} />
            <Route path="/admin/notice/:id" element={<NoticeUpdatePage />} />
            <Route path="/admin/notice/:id/edit" element={<NoticeEditPage />} />
            <Route path="/admin/notice/create" element={<NoticeCreatePage />} />
            <Route path="/admin/qna" element={<QuestionListPage />} />
            <Route path="/admin/qna/:id" element={<AnswerPage />} />
            <Route path="/admin/kpi" element={<KPI />} />
            <Route path="/admin/admin" element={<AdminEdit />} />
        </Routes>
    );
};

export default AdminRoutes;
