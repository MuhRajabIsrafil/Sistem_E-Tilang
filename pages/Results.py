import streamlit as st
from functions import function_system


def app(mycursor):
    st.title("Results Page")

    sql_datetime = "Select datetime from results"
    mycursor.execute(sql_datetime)
    data = mycursor.fetchall()
    unique_data = function_system.fix_array(data)

    results_path = '../results'

    container_result_video = st.container(border=1)

    with container_result_video:
        option_folder = st.selectbox("Select a Folder", unique_data, index=None, placeholder="Select a Folder File...")

        if option_folder:
            image_dir = f'{results_path}/{option_folder}'

            sql_file = f'Select image from results where datetime = "{option_folder}"'
            mycursor.execute(sql_file)

            files = mycursor.fetchall()
            fix_files = function_system.fix_array(files)

            images_path_array = []
            for file in fix_files:
                images_path_array.append(image_dir + '/' + file)

            st.subheader("Results")

            num_columns = 5
            cols_image = st.columns(num_columns)

            if len(images_path_array) == 0:
                no_img_html = """
                <div style="text-align: center"> No Image Result In This Folder </div>
                """

                st.markdown(no_img_html, unsafe_allow_html=True)
            else:
                for i, image_path in enumerate(images_path_array):
                    with cols_image[i % num_columns]:
                        st.image(image_path, caption=f'Result Image {i + 1}', use_column_width=True)
