import React, { useState, useEffect }from 'react';
import { Link } from 'react-router-dom';
import CssBaseline from '@mui/material/CssBaseline';
import AppTheme from '../shared-theme/AppTheme';
import { Box, Typography, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, TablePagination, Button } from '@mui/material';
import MenuContent from '../dashboard/components/MenuContent'
import Header from '../dashboard/components/Header'
import Stack from '@mui/material/Stack';
import { alpha } from '@mui/material/styles';
import axiosInstance from '../api/axiosInstance';

export default function NoticeListPage(props) {
  const [notices, setNotices] = useState([]); // 공지사항 데이터
  const [page, setPage] = useState(0); // Current page number
  const [rowsPerPage, setRowsPerPage] = useState(10); // Number of rows per page

  const fetchNotices = async () => {
    const response = await axiosInstance.get('/notices'); // API 호출
    setNotices(response.data); // 데이터를 상태에 저장
  };

  useEffect(() => {
    fetchNotices();
  }, []);

  // Handle change of page
  const handleChangePage = (event, newPage) => {
    setPage(newPage);
  };

  // Handle change of rows per page
  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0); // Reset to the first page when rows per page changes
  };

  const handleRowClick = (id) => {
    // 클릭한 행의 ID에 해당하는 상세 페이지로 이동
    window.location.href = `/admin/notice/${id}`;
  };

  return (
    <AppTheme {...props}>
          <CssBaseline enableColorScheme />
          <Box sx={{ display: 'flex' }}>
        <MenuContent />
        
        {/* Main content */}
        <Box
          component="main"
          sx={(theme) => ({
            flexGrow: 1,
            backgroundColor: theme.vars
              ? `rgba(${theme.vars.palette.background.defaultChannel} / 1)`
              : alpha(theme.palette.background.default, 1),
            overflow: 'auto',
          })}
        >
          <Stack
            spacing={2}
            sx={{
              alignItems: 'center',
              mx: 3,
              pb: 5,
              mt: { xs: 8, md: 0 },
            }}
          >
            <Header />
             
            <Box sx={{
                maxWidth: 1000,
                width: '100%',
                }}
            >
                <Typography variant="h4" gutterBottom sx={{ marginBottom: 2 }}>
                    공지사항
                </Typography>
                
                <TableContainer component={Paper}>
                    <Table>
                    <TableHead>
                        <TableRow>
                        <TableCell align="left">번호</TableCell>
                        <TableCell align="left">제목</TableCell>
                        <TableCell align="left">날짜</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {notices
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage) // Pagination
                            .map((notice, index) => (
                            <TableRow
                                key={notice.noticeNum}
                                hover
                                sx={{ cursor: 'pointer' }} // 클릭 시 커서 변경
                                onClick={() => handleRowClick(notice.noticeNum)} // 클릭 시 상세 페이지로 이동
                            >
                                <TableCell align="left">{index + 1 + page * rowsPerPage}</TableCell>
                                <TableCell align="left">{notice.noticeTitle}</TableCell>
                                <TableCell align="left">{notice.createdAt}</TableCell>
                            </TableRow>
                        ))}
                    </TableBody>
                    </Table>
                </TableContainer>
                
                <Box 
                    sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        justifyContent: 'space-between', // 자식 요소를 양쪽 끝으로 배치
                        marginTop: 2 
                    }}
                    >
                    {/* TablePagination 가운데 정렬 */}
                    <Box sx={{ display: 'flex', justifyContent: 'center', flex: 1 }}>
                        <TablePagination
                        rowsPerPageOptions={[]} // Options for how many rows per page
                        component="div"
                        count={notices.length} // Total number of notices
                        rowsPerPage={rowsPerPage}
                        page={page}
                        onPageChange={handleChangePage}
                        onRowsPerPageChange={handleChangeRowsPerPage}
                        labelDisplayedRows={() => ''} // Remove "1–7 of 7"
                        />
                    </Box>

                    {/* 작성 버튼 오른쪽 정렬 */}
                    <Box sx={{ flex: 0 }}>
                        <Button 
                        variant="outlined" 
                        color="secondary" 
                        sx={{ marginLeft: 2, marginBottom: 1}} 
                        component={Link} 
                        to="/admin/notice/create"
                        >
                        작성
                        </Button>
                    </Box>
                </Box>
            </Box> 
          </Stack>
        </Box>
      </Box>
           </AppTheme>
  );
}